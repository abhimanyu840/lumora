//! End-to-End Integration and Performance Benchmark Test Suite for Lumora Engine.
//!
//! Verifies the complete photo editing DAG pipeline:
//! Input Buffer -> Exposure + Tone Curves + 8-Channel HSL + 3-Way Color Grading
//!              -> U-Point Lab Pins -> AI Enhancement Node -> Memory-Bounded Tiled Exporter
//!
//! Validates:
//! 1. Exported file headers and magic bytes across JPEG, PNG, and WebP.
//! 2. Tile size invariance and rendering parity.
//! 3. Performance benchmark: processing and exporting a 12MP image in under 1.5 seconds.

use std::time::{Duration, Instant};

use lumora_ai::pipeline::AiZeroDceOp;
use lumora_core::buffer::PixelBufferF32;
use lumora_core::dag::{ContrastOp, ExposureOp, PipelineDAG, SaturationOp};
use lumora_core::exporter::{ExportFormat, ImageExporter};
use lumora_ffi::LumoraImage;
use lumora_ops::color_grading::ColorGrading;
use lumora_ops::curves::{CurvePoint, ToneCurves};
use lumora_ops::details::DetailsAdjustment;
use lumora_ops::hsl::HslMixer;
use lumora_ops::upoint::{UPointMask, UPointPin};

/// Generates a test photo pattern with smooth luminance gradients, color blocks, and highlights.
fn create_test_pattern(width: usize, height: usize) -> PixelBufferF32 {
    let mut buf = PixelBufferF32::new(width, height);
    for y in 0..height {
        let norm_y = y as f32 / height as f32;
        let row_offset = y * width * 4;
        for x in 0..width {
            let norm_x = x as f32 / width as f32;
            let idx = row_offset + x * 4;

            // Gradient background with warm sunset tones
            let r = (norm_x * 0.8 + 0.1).clamp(0.0, 1.0);
            let g = (norm_y * 0.6 + 0.1).clamp(0.0, 1.0);
            let b = ((1.0 - norm_x * 0.5 - norm_y * 0.5) * 0.7).clamp(0.0, 1.0);

            buf.data[idx] = r;
            buf.data[idx + 1] = g;
            buf.data[idx + 2] = b;
            buf.data[idx + 3] = 1.0;
        }
    }
    buf
}

#[test]
fn test_end_to_end_pipeline_integration_and_magic_bytes() {
    let width = 256;
    let height = 256;
    let source = create_test_pattern(width, height);

    // 1. Build the full non-destructive DAG pipeline
    let mut dag = PipelineDAG::new();

    // Exposure (+0.6 stops)
    dag.add_node(ExposureOp::new(0.6));
    // Contrast (+0.15)
    dag.add_node(ContrastOp::new(0.15));
    // Saturation (+0.1)
    dag.add_node(SaturationOp::new(0.1));

    // Lightroom-style S-Curve
    let mut curves = ToneCurves::linear();
    curves.master = vec![
        CurvePoint::new(0.0, 0.0),
        CurvePoint::new(0.25, 0.20), // dip shadows
        CurvePoint::new(0.75, 0.82), // lift highlights
        CurvePoint::new(1.0, 1.0),
    ];
    dag.add_node(curves);

    // 8-channel HSL mixer
    let mut hsl = HslMixer::default();
    hsl.orange.luminance = 0.15; // skin illumination
    hsl.blue.saturation = 0.25; // sky punch
    hsl.green.hue = -10.0; // warmer foliage
    dag.add_node(hsl);

    // 3-way color grading (teal shadows, warm highlights)
    let mut grading = ColorGrading::default();
    grading.shadows.hue = 215.0;
    grading.shadows.saturation = 0.3;
    grading.highlights.hue = 40.0;
    grading.highlights.saturation = 0.25;
    dag.add_node(grading);

    // Snapseed U-Point selective pin
    let mut upoints = UPointMask::new();
    let mut pin = UPointPin::new(0.3, 0.4, 0.25);
    pin.brightness = 0.3;
    pin.contrast = 0.2;
    pin.saturation = 0.2;
    upoints.add_pin(pin);
    upoints.resolve_targets(&source);
    dag.add_node(upoints);

    // AI Zero-DCE low-light enhancement node
    dag.add_node(AiZeroDceOp::new(0.4));

    assert_eq!(dag.node_count(), 8);

    // 2. Export to JPEG using 64x64 tiles
    let exporter_jpeg = ImageExporter::new(ExportFormat::Jpeg { quality: 92 }).with_tile_size(64);
    let jpeg_bytes = exporter_jpeg
        .export_tiled_dag(&source, &dag)
        .expect("JPEG export should succeed");
    assert!(!jpeg_bytes.is_empty());
    // JPEG Magic: 0xFF, 0xD8, 0xFF
    assert_eq!(
        &jpeg_bytes[0..3],
        &[0xFF, 0xD8, 0xFF],
        "JPEG must start with 0xFF 0xD8 0xFF SOI marker"
    );

    // 3. Export to PNG using 64x64 tiles
    let exporter_png = ImageExporter::new(ExportFormat::Png).with_tile_size(64);
    let png_bytes = exporter_png
        .export_tiled_dag(&source, &dag)
        .expect("PNG export should succeed");
    assert!(!png_bytes.is_empty());
    // PNG Magic: 0x89 0x50 0x4E 0x47 0x0D 0x0A 0x1A 0x0A
    assert_eq!(
        &png_bytes[0..8],
        &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A],
        "PNG must start with standard 8-byte signature"
    );

    // 4. Export to WebP using 64x64 tiles
    let exporter_webp =
        ImageExporter::new(ExportFormat::WebP { lossless: true }).with_tile_size(64);
    let webp_bytes = exporter_webp
        .export_tiled_dag(&source, &dag)
        .expect("WebP export should succeed");
    assert!(!webp_bytes.is_empty());
    // WebP Magic: "RIFF" .... "WEBP"
    assert_eq!(&webp_bytes[0..4], b"RIFF");
    assert_eq!(&webp_bytes[8..12], b"WEBP");
}

#[test]
fn test_tiled_export_parity_full_pipeline() {
    let width = 256;
    let height = 192;
    let source = create_test_pattern(width, height);

    let mut dag = PipelineDAG::new();
    dag.add_node(ExposureOp::new(0.4));
    dag.add_node(ContrastOp::new(0.1));
    dag.add_node(SaturationOp::new(0.15));

    let mut hsl = HslMixer::default();
    hsl.red.saturation = 0.2;
    hsl.blue.hue = 15.0;
    dag.add_node(hsl);

    let mut upoints = UPointMask::new();
    let mut pin = UPointPin::new(0.5, 0.5, 0.3);
    pin.brightness = 0.2;
    upoints.add_pin(pin);
    upoints.resolve_targets(&source);
    dag.add_node(upoints);

    // Render with 64x64 tiles
    let exp_64 = ImageExporter::new(ExportFormat::Png).with_tile_size(64);
    let rendered_64 = exp_64.render_tiled_dag(&source, &dag).unwrap();

    // Render with 128x128 tiles
    let exp_128 = ImageExporter::new(ExportFormat::Png).with_tile_size(128);
    let rendered_128 = exp_128.render_tiled_dag(&source, &dag).unwrap();

    // Render with 512x512 tiles (exceeds dimensions, single tile)
    let exp_512 = ImageExporter::new(ExportFormat::Png).with_tile_size(512);
    let rendered_512 = exp_512.render_tiled_dag(&source, &dag).unwrap();

    assert_eq!(rendered_64.data.len(), rendered_512.data.len());
    assert_eq!(rendered_128.data.len(), rendered_512.data.len());

    // Verify mathematical identity across tile boundaries
    assert_eq!(
        rendered_64.data, rendered_512.data,
        "64x64 tiled rendering must exactly equal 512x512 single-tile rendering"
    );
    assert_eq!(
        rendered_128.data, rendered_512.data,
        "128x128 tiled rendering must exactly equal 512x512 single-tile rendering"
    );
}

#[test]
fn test_end_to_end_benchmark_performance_under_1_5s() {
    // 12 Megapixel benchmark image: 4000 x 3000 pixels
    let width = 4000;
    let height = 3000;
    let total_pixels = width * height;

    println!("Constructing 12MP ({width}x{height} = {total_pixels} pixels) test image...");
    let source = create_test_pattern(width, height);

    // Build the full production pipeline DAG:
    // Exposure + Curves + HSL + 3-Way Color Grading + U-Point Mask + AI Zero-DCE
    let mut dag = PipelineDAG::new();

    // 1. Exposure (+0.5 stops)
    dag.add_node(ExposureOp::new(0.5));

    // 2. S-Curve
    let mut curves = ToneCurves::linear();
    curves.master = vec![
        CurvePoint::new(0.0, 0.0),
        CurvePoint::new(0.25, 0.22),
        CurvePoint::new(0.75, 0.80),
        CurvePoint::new(1.0, 1.0),
    ];
    dag.add_node(curves);

    // 3. 8-Channel HSL Mixer
    let mut hsl = HslMixer::default();
    hsl.orange.saturation = 0.1;
    hsl.blue.saturation = 0.2;
    dag.add_node(hsl);

    // 4. 3-Way Color Grading
    let mut grading = ColorGrading::default();
    grading.shadows.hue = 210.0;
    grading.shadows.saturation = 0.2;
    grading.highlights.hue = 45.0;
    grading.highlights.saturation = 0.15;
    dag.add_node(grading);

    // 5. U-Point Control Pin
    let mut upoints = UPointMask::new();
    let mut pin = UPointPin::new(0.5, 0.5, 0.3);
    pin.brightness = 0.25;
    pin.contrast = 0.1;
    upoints.add_pin(pin);
    upoints.resolve_targets(&source);
    dag.add_node(upoints);

    // 6. AI Zero-DCE low-light enhancement node
    dag.add_node(AiZeroDceOp::new(0.5));

    assert_eq!(dag.node_count(), 6);

    // 512x512 tiled exporter to JPEG
    let exporter = ImageExporter::new(ExportFormat::Jpeg { quality: 85 }).with_tile_size(512);

    println!("Starting full-pipeline 12MP tiled export benchmark...");
    let start_time = Instant::now();

    let jpeg_bytes = exporter
        .export_tiled_dag(&source, &dag)
        .expect("Tiled DAG export must succeed");

    let elapsed = start_time.elapsed();
    let mpixels_per_sec = (total_pixels as f64 / 1_000_000.0) / elapsed.as_secs_f64();

    println!(
        "Benchmark Result: 12MP full-pipeline export completed in {:.3} ms ({:.2} MP/s). Output size: {:.2} MB",
        elapsed.as_secs_f64() * 1000.0,
        mpixels_per_sec,
        jpeg_bytes.len() as f64 / (1024.0 * 1024.0)
    );

    // Assert file correctness
    assert!(!jpeg_bytes.is_empty());
    assert_eq!(&jpeg_bytes[0..3], &[0xFF, 0xD8, 0xFF]);

    // Hard ship-gate requirement: full end-to-end 12MP processing < 1.5 seconds (1500 ms) in release mode
    if !cfg!(debug_assertions) {
        assert!(
            elapsed < Duration::from_millis(1500),
            "Performance regression: 12MP full-pipeline export took {:?}, exceeding 1500ms benchmark target!",
            elapsed
        );
    } else {
        // In unoptimized debug builds, simply verify execution completes and output is valid
        println!("(Debug mode: timing check bypassed; run with --release for strict 1.5s ship-gate verification)");
    }
}

#[test]
fn test_ffi_session_e2e_export() {
    let width = 128u32;
    let height = 128u32;
    let raw_rgba = vec![120u8; (width * height * 4) as usize];

    let mut session = LumoraImage::new(width, height, &raw_rgba).expect("Create session");
    session.exposure = 0.4;
    session.contrast = 0.2;
    session.saturation = 0.15;
    session.color_grading.shadows.hue = 200.0;
    session.color_grading.shadows.saturation = 0.3;

    let mut pin = UPointPin::new(0.5, 0.5, 0.3);
    pin.brightness = 0.2;
    session.upoints.add_pin(pin);

    // Add Zero-DCE AI pass
    session.ai_passes.push((2, 0.5));

    // Export via session helper
    let jpeg_bytes = session
        .export_full_resolution(ExportFormat::Jpeg { quality: 90 }, 64)
        .expect("Session export full resolution");

    assert!(!jpeg_bytes.is_empty());
    assert_eq!(&jpeg_bytes[0..3], &[0xFF, 0xD8, 0xFF]);
}

#[test]
fn test_tiled_export_parity_unresolved_upoint_and_details() {
    let width = 128;
    let height = 96;
    let source = create_test_pattern(width, height);

    let mut dag = PipelineDAG::new();
    dag.add_node(ExposureOp::new(0.3));

    // Spatial filter: Details sharpening across tile seams
    dag.add_node(DetailsAdjustment::new(0.25, 0.4));

    // Spatial filter: U-Point pin WITHOUT pre-resolving targets (relies on source sampling across tiles)
    let mut upoints = UPointMask::new();
    let mut pin = UPointPin::new(0.35, 0.35, 0.4);
    pin.brightness = 0.3;
    pin.contrast = 0.2;
    upoints.add_pin(pin);
    dag.add_node(upoints);

    // 32x32 tiles (many tile seams crossing pin and details filter)
    let exp_32 = ImageExporter::new(ExportFormat::Png).with_tile_size(32);
    let rendered_32 = exp_32.render_tiled_dag(&source, &dag).unwrap();

    // 64x64 tiles
    let exp_64 = ImageExporter::new(ExportFormat::Png).with_tile_size(64);
    let rendered_64 = exp_64.render_tiled_dag(&source, &dag).unwrap();

    // Untiled 512x512
    let exp_untiled = ImageExporter::new(ExportFormat::Png).with_tile_size(512);
    let rendered_untiled = exp_untiled.render_tiled_dag(&source, &dag).unwrap();

    assert_eq!(
        rendered_32.data, rendered_untiled.data,
        "32x32 tiled render with UPoint and Details must match untiled render identically"
    );
    assert_eq!(
        rendered_64.data, rendered_untiled.data,
        "64x64 tiled render with UPoint and Details must match untiled render identically"
    );
}

#[test]
fn test_super_res_session_export_4x_dimensions() {
    let width = 32u32;
    let height = 24u32;
    let raw_rgba = vec![140u8; (width * height * 4) as usize];

    let mut session = LumoraImage::new(width, height, &raw_rgba).expect("Create session");
    session.exposure = 0.2;
    // Add Super-Resolution AI pass (op_type == 3)
    session.ai_passes.push((3, 1.0));

    let png_bytes = session
        .export_full_resolution(ExportFormat::Png, 16)
        .expect("Export super-res PNG");

    assert!(!png_bytes.is_empty());
    assert_eq!(
        &png_bytes[0..8],
        &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]
    );

    // Read IHDR chunk dimensions (width at bytes 16..20, height at 20..24 in big-endian)
    let out_w = u32::from_be_bytes(png_bytes[16..20].try_into().unwrap());
    let out_h = u32::from_be_bytes(png_bytes[20..24].try_into().unwrap());
    assert_eq!(
        out_w,
        width * 4,
        "Super-resolution export width must be 4x original"
    );
    assert_eq!(
        out_h,
        height * 4,
        "Super-resolution export height must be 4x original"
    );
}
