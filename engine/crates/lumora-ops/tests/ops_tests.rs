use lumora_core::buffer::{PixelBufferF32, RgbaF32};
use lumora_core::dag::{ExposureOp, PipelineDAG};
use lumora_ops::color_grading::ColorGrading;
use lumora_ops::curves::{evaluate_spline, CurvePoint, ToneCurves};
use lumora_ops::details::DetailsAdjustment;
use lumora_ops::hsl::HslMixer;
use lumora_ops::presets::{CubeLut, LumoraPreset, XmpPreset};
use lumora_ops::upoint::{UPointMask, UPointPin};
use std::time::Instant;

#[test]
fn test_cubic_spline_monotonicity() {
    // S-curve points
    let points = vec![
        CurvePoint::new(0.0, 0.0),
        CurvePoint::new(0.25, 0.15),
        CurvePoint::new(0.5, 0.5),
        CurvePoint::new(0.75, 0.85),
        CurvePoint::new(1.0, 1.0),
    ];

    // Check monotonic increase across 100 sample points
    let mut prev_y = -1.0;
    for i in 0..=100 {
        let x = i as f32 / 100.0;
        let y = evaluate_spline(&points, x);
        assert!(
            y >= prev_y - 1e-6,
            "Monotonicity violated at x={x}: y={y} < prev={prev_y}"
        );
        assert!(
            (0.0..=1.0).contains(&y),
            "Output out of [0, 1] range: y={y}"
        );
        prev_y = y;
    }
}

#[test]
fn test_curves_unsorted_and_duplicate_points() {
    // Points intentionally out of order with duplicate x coordinates
    let raw_points = vec![
        CurvePoint::new(1.0, 1.0),
        CurvePoint::new(0.0, 0.0),
        CurvePoint::new(0.5, 0.7),
        CurvePoint::new(0.5, 0.75), // Duplicate x
        CurvePoint::new(0.2, 0.3),
    ];

    let y_mid = evaluate_spline(&raw_points, 0.5);
    assert!(
        (y_mid - 0.75).abs() < 1e-4,
        "Expected y at 0.5 to be 0.75: got {y_mid}"
    );

    let y_zero = evaluate_spline(&raw_points, 0.0);
    assert!((y_zero - 0.0).abs() < 1e-4);

    let y_one = evaluate_spline(&raw_points, 1.0);
    assert!((y_one - 1.0).abs() < 1e-4);
}

#[test]
fn test_tone_curves_application() {
    let mut buf = PixelBufferF32::new(4, 4);
    for chunk in buf.data.chunks_exact_mut(4) {
        chunk[0] = 0.25;
        chunk[1] = 0.50;
        chunk[2] = 0.75;
        chunk[3] = 1.0;
    }

    let mut curves = ToneCurves::linear();
    // Darken shadows on master
    curves.master = vec![
        CurvePoint::new(0.0, 0.0),
        CurvePoint::new(0.25, 0.10),
        CurvePoint::new(0.5, 0.5),
        CurvePoint::new(1.0, 1.0),
    ];
    curves.apply_to_buffer(&mut buf);

    let p = buf.get_pixel(0, 0).unwrap();
    // 0.25 was remapped to ~0.10
    assert!(p.r < 0.20, "Expected darkened red: got {}", p.r);
}

#[test]
fn test_hsl_mixer_selective_shift() {
    let mut buf = PixelBufferF32::new(2, 2);
    // Pixel 0: Pure Red (H=0, S=1, L=0.5)
    buf.set_pixel(0, 0, RgbaF32::new(1.0, 0.0, 0.0, 1.0))
        .unwrap();
    // Pixel 1: Pure Green (H=120, S=1, L=0.5)
    buf.set_pixel(1, 0, RgbaF32::new(0.0, 1.0, 0.0, 1.0))
        .unwrap();

    let mut hsl = HslMixer::default();
    // Shift Red channel hue by +30 degrees (towards Orange)
    hsl.red.hue = 30.0;
    // Boost Red luminance
    hsl.red.luminance = 0.2;

    hsl.apply_to_buffer(&mut buf);

    let p_red = buf.get_pixel(0, 0).unwrap();
    let p_green = buf.get_pixel(1, 0).unwrap();

    // Red pixel should now have green component (shifted towards Orange)
    assert!(
        p_red.g > 0.05,
        "Expected green component in shifted red: {}",
        p_red.g
    );

    // Green pixel should be completely untouched
    assert_eq!(p_green.r, 0.0);
    assert_eq!(p_green.g, 1.0);
    assert_eq!(p_green.b, 0.0);
}

#[test]
fn test_color_grading_split_toning() {
    let mut buf = PixelBufferF32::new(2, 1);
    // Dark pixel (Shadow)
    buf.set_pixel(0, 0, RgbaF32::new(0.1, 0.1, 0.1, 1.0))
        .unwrap();
    // Bright pixel (Highlight)
    buf.set_pixel(1, 0, RgbaF32::new(0.9, 0.9, 0.9, 1.0))
        .unwrap();

    let mut grading = ColorGrading::default();
    // Tint shadows blue (Hue = 240)
    grading.shadows.hue = 240.0;
    grading.shadows.saturation = 0.8;

    // Tint highlights orange/warm (Hue = 35)
    grading.highlights.hue = 35.0;
    grading.highlights.saturation = 0.8;

    grading.apply_to_buffer(&mut buf);

    let p_shadow = buf.get_pixel(0, 0).unwrap();
    let p_highlight = buf.get_pixel(1, 0).unwrap();

    // Shadow pixel should have more blue than red/green
    assert!(
        p_shadow.b > p_shadow.r,
        "Shadow not tinted blue: {:?}",
        p_shadow
    );

    // Highlight pixel should have more red than blue
    assert!(
        p_highlight.r > p_highlight.b,
        "Highlight not tinted warm: {:?}",
        p_highlight
    );
}

#[test]
fn test_upoint_lab_weighting() {
    let mut buf = PixelBufferF32::new(20, 20);
    // Fill entire canvas with blue sky
    for chunk in buf.data.chunks_exact_mut(4) {
        chunk[0] = 0.2;
        chunk[1] = 0.5;
        chunk[2] = 0.9;
        chunk[3] = 1.0;
    }
    // Create an orange flower patch at bottom right (15, 15)
    for y in 14..=16 {
        for x in 14..=16 {
            buf.set_pixel(x, y, RgbaF32::new(0.9, 0.4, 0.1, 1.0))
                .unwrap();
        }
    }

    // Place U-Point pin on the orange flower at (0.75, 0.75)
    let mut mask = UPointMask::new();
    let mut pin = UPointPin::new(0.75, 0.75, 0.3);
    pin.brightness = 0.4; // Boost flower brightness
    mask.add_pin(pin);

    mask.apply_to_buffer(&mut buf);

    // The orange flower pixel at (15, 15) should be noticeably brightened
    let flower_after = buf.get_pixel(15, 15).unwrap();
    assert!(flower_after.r > 0.95);

    // The sky at (2, 2) has a completely different color and is outside the radius:
    // It should remain unaffected by the flower pin
    let sky_after = buf.get_pixel(2, 2).unwrap();
    assert!((sky_after.r - 0.2).abs() < 0.05);
}

#[test]
fn test_upoint_multi_pin_accumulation() {
    let mut buf = PixelBufferF32::new(10, 10);
    // Fill with gray
    for chunk in buf.data.chunks_exact_mut(4) {
        chunk[0] = 0.5;
        chunk[1] = 0.5;
        chunk[2] = 0.5;
        chunk[3] = 1.0;
    }

    // Add two pins targeting the center: one for brightness, one for contrast
    let mut mask = UPointMask::new();
    let mut pin1 = UPointPin::new(0.5, 0.5, 0.4);
    pin1.brightness = 0.3;
    mask.add_pin(pin1);

    let mut pin2 = UPointPin::new(0.5, 0.5, 0.4);
    pin2.contrast = 0.5;
    mask.add_pin(pin2);

    mask.apply_to_buffer(&mut buf);

    let p = buf.get_pixel(5, 5).unwrap();
    // Both brightness and contrast should be reflected (not overwritten)
    assert!(p.r > 0.55, "Pin 1 brightness was lost: {}", p.r);
}

#[test]
fn test_details_sharpening() {
    let mut buf = PixelBufferF32::new(8, 8);
    // Flat field
    for chunk in buf.data.chunks_exact_mut(4) {
        chunk[0] = 0.5;
        chunk[1] = 0.5;
        chunk[2] = 0.5;
        chunk[3] = 1.0;
    }
    // High-contrast edge at (4, 4)
    buf.set_pixel(4, 4, RgbaF32::new(0.9, 0.9, 0.9, 1.0))
        .unwrap();

    let details = DetailsAdjustment::new(0.5, 0.8);
    details.apply_to_buffer(&mut buf);

    let p_center = buf.get_pixel(4, 4).unwrap();
    // High-pass sharpening increases edge peak
    assert!(p_center.r > 0.9);
}

#[test]
fn test_presets_lumora_json() {
    let mut preset = LumoraPreset::new("Nordic Minimal", "LumoraLab");
    preset.exposure = Some(0.25);
    preset.contrast = Some(0.1);

    let json = preset.to_json().expect("serialize preset");
    assert!(json.contains("Nordic Minimal"));

    let loaded = LumoraPreset::from_json(&json).expect("deserialize preset");
    assert_eq!(loaded.name, "Nordic Minimal");
    assert_eq!(loaded.exposure, Some(0.25));

    let mut buf = PixelBufferF32::new(4, 4);
    for chunk in buf.data.chunks_exact_mut(4) {
        chunk[0] = 0.5;
        chunk[1] = 0.5;
        chunk[2] = 0.5;
        chunk[3] = 1.0;
    }
    loaded.apply_to_buffer(&mut buf);
    assert!(buf.get_pixel(0, 0).unwrap().r > 0.5);
}

#[test]
fn test_presets_cube_lut() {
    // 2x2x2 identity .cube string
    let cube_str = r#"
TITLE "Test Identity 2x2x2"
LUT_3D_SIZE 2
DOMAIN_MIN 0.0 0.0 0.0
DOMAIN_MAX 1.0 1.0 1.0
0.0 0.0 0.0
1.0 0.0 0.0
0.0 1.0 0.0
1.0 1.0 0.0
0.0 0.0 1.0
1.0 0.0 1.0
0.0 1.0 1.0
1.0 1.0 1.0
"#;
    let lut = CubeLut::parse(cube_str).expect("parse cube lut");
    assert_eq!(lut.size, 2);
    assert_eq!(lut.title, "Test Identity 2x2x2");

    let sampled = lut.sample_trilinear(0.5, 0.5, 0.5);
    assert!((sampled.0 - 0.5).abs() < 1e-3);
    assert!((sampled.1 - 0.5).abs() < 1e-3);
    assert!((sampled.2 - 0.5).abs() < 1e-3);
}

#[test]
fn test_presets_xmp_parsing() {
    let xmp_str = r#"
<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"
   crs:Exposure2012="+0.40"
   crs:Contrast2012="+15"
   crs:Saturation="-5"
   crs:ToneCurvePV2012="[0, 0, 64, 55, 192, 200, 255, 255]"
   crs:HueAdjustmentRed="+10"
   crs:SaturationAdjustmentRed="+20"
   crs:LuminanceAdjustmentRed="-10">
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
"#;
    let xmp = XmpPreset::parse_xmp(xmp_str).expect("parse xmp");
    assert_eq!(xmp.exposure, Some(0.40));
    assert_eq!(xmp.contrast, Some(0.15));
    assert_eq!(xmp.saturation, Some(-0.05));
    assert!(xmp.tone_curve.is_some());
    assert_eq!(xmp.tone_curve.as_ref().unwrap().len(), 4);
    assert_eq!(xmp.hsl.red.hue, 10.0);
    assert_eq!(xmp.hsl.red.saturation, 0.20);
    assert_eq!(xmp.hsl.red.luminance, -0.10);
}

#[test]
fn test_benchmark_rendering_speed() {
    // 3.0 MP realistic high-res benchmark chunk (2000 x 1500)
    let width = 2000;
    let height = 1500;
    let mut buf = PixelBufferF32::new(width, height);

    // Generate gradient test pattern
    for y in 0..height {
        let ny = y as f32 / height as f32;
        for x in 0..width {
            let nx = x as f32 / width as f32;
            let idx = (y * width + x) * 4;
            buf.data[idx] = nx;
            buf.data[idx + 1] = ny;
            buf.data[idx + 2] = 1.0 - (nx + ny) * 0.5;
            buf.data[idx + 3] = 1.0;
        }
    }

    let mut dag = PipelineDAG::new();
    dag.add_node(ExposureOp::new(0.3));

    let mut curves = ToneCurves::linear();
    curves.master = vec![
        CurvePoint::new(0.0, 0.0),
        CurvePoint::new(0.5, 0.55),
        CurvePoint::new(1.0, 1.0),
    ];
    dag.add_node(curves);

    let mut hsl = HslMixer::default();
    hsl.red.hue = 15.0;
    hsl.blue.saturation = 0.2;
    dag.add_node(hsl);

    let mut grading = ColorGrading::default();
    grading.shadows.hue = 220.0;
    grading.shadows.saturation = 0.3;
    dag.add_node(grading);

    // Full U-Point selective mask node with 2 active pins
    let mut upoint = UPointMask::new();
    let mut pin1 = UPointPin::new(0.3, 0.3, 0.25);
    pin1.brightness = 0.2;
    upoint.add_pin(pin1);
    let mut pin2 = UPointPin::new(0.7, 0.7, 0.25);
    pin2.saturation = 0.3;
    upoint.add_pin(pin2);
    dag.add_node(upoint);

    let start = Instant::now();
    let rendered = dag.render(&buf);
    let elapsed = start.elapsed();

    assert_eq!(rendered.dimensions(), (width, height));
    println!(
        "Processed 3MP image with full 5-node DAG (including U-Point) in {:?}",
        elapsed
    );
    // Must be fast and responsive: < 350ms in debug, < 30ms in release
    assert!(
        elapsed.as_millis() < 350,
        "Rendering too slow: {:?}",
        elapsed
    );
}
