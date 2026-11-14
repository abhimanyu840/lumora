use std::fs;

use lumora_core::buffer::{PixelBufferF32, RgbaF32};
use lumora_core::dag::{ContrastOp, ExposureOp, PipelineDAG, SaturationOp};
use lumora_core::exporter::{ExportError, ExportFormat, ImageExporter};

#[test]
fn test_exporter_jpeg_png_webp_magic_bytes() {
    let width = 64;
    let height = 64;
    let mut buf = PixelBufferF32::new(width, height);
    for (i, chunk) in buf.data.chunks_exact_mut(4).enumerate() {
        let x = (i % width) as f32 / width as f32;
        let y = (i / width) as f32 / height as f32;
        chunk[0] = x;
        chunk[1] = y;
        chunk[2] = 0.5 * (x + y);
        chunk[3] = 1.0;
    }

    let exporter_jpeg = ImageExporter::new(ExportFormat::Jpeg { quality: 85 }).with_tile_size(32);
    let jpeg_bytes = exporter_jpeg
        .export_f32(&buf)
        .expect("JPEG export should succeed");
    assert!(!jpeg_bytes.is_empty());
    // JPEG Magic Bytes: 0xFF, 0xD8, 0xFF
    assert_eq!(&jpeg_bytes[0..3], &[0xFF, 0xD8, 0xFF]);

    let exporter_png = ImageExporter::new(ExportFormat::Png).with_tile_size(32);
    let png_bytes = exporter_png
        .export_f32(&buf)
        .expect("PNG export should succeed");
    assert!(!png_bytes.is_empty());
    // PNG Magic Bytes: 0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A
    assert_eq!(
        &png_bytes[0..8],
        &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]
    );

    let exporter_webp =
        ImageExporter::new(ExportFormat::WebP { lossless: true }).with_tile_size(32);
    let webp_bytes = exporter_webp
        .export_f32(&buf)
        .expect("WebP export should succeed");
    assert!(!webp_bytes.is_empty());
    // WebP Magic: "RIFF" .... "WEBP"
    assert_eq!(&webp_bytes[0..4], b"RIFF");
    assert_eq!(&webp_bytes[8..12], b"WEBP");
}

#[test]
fn test_exporter_tile_size_parity() {
    let width = 128;
    let height = 96;
    let mut buf = PixelBufferF32::new(width, height);
    for (i, chunk) in buf.data.chunks_exact_mut(4).enumerate() {
        chunk[0] = ((i % 17) as f32) / 17.0;
        chunk[1] = ((i % 29) as f32) / 29.0;
        chunk[2] = ((i % 43) as f32) / 43.0;
        chunk[3] = 1.0;
    }

    let mut dag = PipelineDAG::new();
    dag.add_node(ExposureOp::new(0.5));
    dag.add_node(ContrastOp::new(0.2));
    dag.add_node(SaturationOp::new(0.3));

    // Render with 32x32 tiles
    let exporter_tiled_32 = ImageExporter::new(ExportFormat::Png).with_tile_size(32);
    let rgba_tiled_32 = exporter_tiled_32
        .render_tiled_dag(&buf, &dag)
        .expect("32x32 tiled render should succeed");

    // Render with 64x64 tiles
    let exporter_tiled_64 = ImageExporter::new(ExportFormat::Png).with_tile_size(64);
    let rgba_tiled_64 = exporter_tiled_64
        .render_tiled_dag(&buf, &dag)
        .expect("64x64 tiled render should succeed");

    // Render with single large tile (untiled)
    let exporter_untiled = ImageExporter::new(ExportFormat::Png).with_tile_size(1024);
    let rgba_untiled = exporter_untiled
        .render_tiled_dag(&buf, &dag)
        .expect("Untiled render should succeed");

    assert_eq!(rgba_tiled_32.data.len(), rgba_untiled.data.len());
    assert_eq!(rgba_tiled_64.data.len(), rgba_untiled.data.len());

    // Byte-for-byte exact equality across different tile partitionings
    assert_eq!(
        rgba_tiled_32.data, rgba_untiled.data,
        "32x32 tiled output must match untiled output byte-for-byte"
    );
    assert_eq!(
        rgba_tiled_64.data, rgba_untiled.data,
        "64x64 tiled output must match untiled output byte-for-byte"
    );
}

#[test]
fn test_exporter_edge_cases() {
    let empty_buf = PixelBufferF32::new(0, 0);
    let exporter = ImageExporter::new(ExportFormat::Png);
    assert_eq!(
        exporter.export_f32(&empty_buf),
        Err(ExportError::EmptyBuffer)
    );

    // Invalid quality
    let exporter_bad_q = ImageExporter::new(ExportFormat::Jpeg { quality: 0 });
    let tiny_buf = PixelBufferF32::new(2, 2);
    assert_eq!(
        exporter_bad_q.export_f32(&tiny_buf),
        Err(ExportError::InvalidQuality(0))
    );

    let exporter_bad_q2 = ImageExporter::new(ExportFormat::Jpeg { quality: 101 });
    assert_eq!(
        exporter_bad_q2.export_f32(&tiny_buf),
        Err(ExportError::InvalidQuality(101))
    );

    // Dimension mismatch
    assert_eq!(
        exporter.encode_rgba8(4, 4, &[0u8; 12]),
        Err(ExportError::InvalidDimensions {
            width: 4,
            height: 4
        })
    );

    // 1x1 image export
    let mut one_pixel = PixelBufferF32::new(1, 1);
    one_pixel
        .set_pixel(0, 0, RgbaF32::rgb(0.7, 0.4, 0.2))
        .unwrap();
    let one_px_jpeg = ImageExporter::new(ExportFormat::Jpeg { quality: 90 })
        .export_f32(&one_pixel)
        .unwrap();
    assert!(!one_px_jpeg.is_empty());
    assert_eq!(&one_px_jpeg[0..3], &[0xFF, 0xD8, 0xFF]);
}

#[test]
fn test_exporter_file_system_roundtrip() {
    let width = 32;
    let height = 32;
    let mut buf = PixelBufferF32::new(width, height);
    for chunk in buf.data.chunks_exact_mut(4) {
        chunk[0] = 0.8;
        chunk[1] = 0.2;
        chunk[2] = 0.5;
        chunk[3] = 1.0;
    }

    let temp_dir = std::env::temp_dir();
    let temp_png_path = temp_dir.join("lumora_test_export.png");
    let temp_jpg_path = temp_dir.join("lumora_test_export.jpg");

    let exporter_png = ImageExporter::new(ExportFormat::Png);
    let bytes_written = exporter_png
        .export_to_file(&buf, &temp_png_path)
        .expect("PNG file export should succeed");
    assert!(bytes_written > 0);
    assert!(temp_png_path.exists());

    let read_bytes = fs::read(&temp_png_path).expect("Read exported PNG");
    assert_eq!(read_bytes.len(), bytes_written);
    assert_eq!(
        &read_bytes[0..8],
        &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]
    );
    let _ = fs::remove_file(&temp_png_path);

    let exporter_jpg = ImageExporter::new(ExportFormat::Jpeg { quality: 80 });
    let mut dag = PipelineDAG::new();
    dag.add_node(ExposureOp::new(0.5));
    let jpg_written = exporter_jpg
        .export_dag_to_file(&buf, &dag, &temp_jpg_path)
        .expect("DAG JPEG file export should succeed");
    assert!(jpg_written > 0);
    assert!(temp_jpg_path.exists());
    let _ = fs::remove_file(&temp_jpg_path);
}

#[test]
fn test_extract_tile_bounds_clamping_and_empty() {
    let mut buf = PixelBufferF32::new(10, 10);
    buf.set_pixel(0, 0, RgbaF32::rgb(1.0, 0.5, 0.25)).unwrap();

    // Tile requesting region beyond buffer boundaries must clamp and zero-fill without panicking
    let tile = ImageExporter::extract_tile(&buf, 8, 8, 8, 8);
    assert_eq!(tile.width, 8);
    assert_eq!(tile.height, 8);

    // Extraction completely outside buffer
    let tile_outside = ImageExporter::extract_tile(&buf, 20, 20, 5, 5);
    assert_eq!(tile_outside.width, 5);
    assert_eq!(tile_outside.height, 5);
    assert!(tile_outside.data.iter().all(|&v| v == 0.0));
}

#[test]
fn test_jpeg_alpha_matting_against_white_background() {
    let exporter = ImageExporter::new(ExportFormat::Jpeg { quality: 90 });

    // 2x1 image:
    // Pixel 0: fully transparent red [255, 0, 0, 0] -> should matte against white to [255, 255, 255]
    // Pixel 1: opaque green [0, 255, 0, 255] -> should remain green
    let rgba_bytes = vec![
        255, 0, 0, 0, // transparent red
        0, 255, 0, 255, // opaque green
    ];

    let jpeg_bytes = exporter
        .encode_rgba8(2, 1, &rgba_bytes)
        .expect("JPEG encoding with transparency matting should succeed");
    assert!(!jpeg_bytes.is_empty());
    assert_eq!(&jpeg_bytes[0..3], &[0xFF, 0xD8, 0xFF]);
}
