use lumora_core::buffer::{PixelBufferF32, PixelBufferRgba8, Rgba8};
use lumora_core::color::{
    hsl_to_rgb, lab_to_srgb, linear_to_srgb, rgb_to_hsl, srgb_to_lab, srgb_to_linear, Lab,
};
use lumora_core::dag::{ContrastOp, ExposureOp, PipelineDAG, SaturationOp};

#[test]
fn test_rgba8_buffer_lifecycle() {
    let mut buf = PixelBufferRgba8::new(10, 10);
    assert_eq!(buf.dimensions(), (10, 10));
    assert_eq!(buf.pixel_count(), 100);

    // Initial buffer should be black
    let p0 = buf.get_pixel(0, 0).expect("pixel 0,0");
    assert_eq!(p0, Rgba8::new(0, 0, 0, 0));

    // Set pixel
    let test_pixel = Rgba8::new(255, 128, 64, 255);
    buf.set_pixel(5, 5, test_pixel).expect("set pixel");
    assert_eq!(buf.get_pixel(5, 5), Some(test_pixel));

    // Out of bounds check
    assert!(buf.get_pixel(10, 10).is_none());
    assert!(buf.set_pixel(10, 5, test_pixel).is_err());
}

#[test]
fn test_buffer_f32_conversion_roundtrip() {
    let mut buf_u8 = PixelBufferRgba8::new(4, 4);
    buf_u8
        .set_pixel(1, 1, Rgba8::new(200, 100, 50, 255))
        .unwrap();

    let buf_f32 = buf_u8.to_f32();
    let p_f32 = buf_f32.get_pixel(1, 1).unwrap();
    assert!((p_f32.r - 200.0 / 255.0).abs() < 1e-4);
    assert!((p_f32.g - 100.0 / 255.0).abs() < 1e-4);
    assert!((p_f32.b - 50.0 / 255.0).abs() < 1e-4);
    assert!((p_f32.a - 1.0).abs() < 1e-4);

    let roundtrip_u8 = buf_f32.to_rgba8();
    assert_eq!(roundtrip_u8.get_pixel(1, 1), buf_u8.get_pixel(1, 1));
}

#[test]
fn test_color_srgb_linear_roundtrip() {
    let samples = [0.0f32, 0.04, 0.18, 0.5, 0.75, 1.0];
    for &s in &samples {
        let lin = srgb_to_linear(s);
        let back = linear_to_srgb(lin);
        assert!(
            (s - back).abs() < 1e-4,
            "srgb-linear mismatch for {s}: got {back}"
        );
    }
}

#[test]
fn test_color_srgb_lab_roundtrip() {
    // Pure Red, Green, Blue, White, Black, Mid-gray
    let colors = [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 1.0, 1.0),
        (0.0, 0.0, 0.0),
        (0.5, 0.5, 0.5),
    ];

    for (r, g, b) in colors {
        let lab = srgb_to_lab(r, g, b);
        let (rr, gg, bb) = lab_to_srgb(lab);
        assert!(
            (r - rr).abs() < 0.02,
            "Red mismatch for ({r},{g},{b}): got {rr}"
        );
        assert!(
            (g - gg).abs() < 0.02,
            "Green mismatch for ({r},{g},{b}): got {gg}"
        );
        assert!(
            (b - bb).abs() < 0.02,
            "Blue mismatch for ({r},{g},{b}): got {bb}"
        );
    }
}

#[test]
fn test_color_hsl_roundtrip() {
    let colors = [
        (1.0, 0.0, 0.0), // Red: H=0
        (0.0, 1.0, 0.0), // Green: H=120
        (0.0, 0.0, 1.0), // Blue: H=240
        (1.0, 1.0, 0.0), // Yellow: H=60
        (0.0, 1.0, 1.0), // Cyan: H=180
        (1.0, 0.0, 1.0), // Magenta: H=300
    ];

    for (r, g, b) in colors {
        let hsl = rgb_to_hsl(r, g, b);
        let (rr, gg, bb) = hsl_to_rgb(hsl);
        assert!((r - rr).abs() < 1e-4);
        assert!((g - gg).abs() < 1e-4);
        assert!((b - bb).abs() < 1e-4);
    }
}

#[test]
fn test_lab_delta_e() {
    let lab1 = Lab::new(50.0, 10.0, -20.0);
    let lab2 = Lab::new(50.0, 10.0, -20.0);
    assert_eq!(lab1.delta_e_cie76(lab2), 0.0);

    let lab3 = Lab::new(53.0, 14.0, -20.0);
    // dl=3, da=4, db=0 -> delta E = 5.0
    assert!((lab1.delta_e_cie76(lab3) - 5.0).abs() < 1e-5);
}

#[test]
fn test_pipeline_dag_operations() {
    let mut buf = PixelBufferF32::new(10, 10);
    for chunk in buf.data.chunks_exact_mut(4) {
        chunk[0] = 0.5;
        chunk[1] = 0.5;
        chunk[2] = 0.5;
        chunk[3] = 1.0;
    }

    let mut dag = PipelineDAG::new();
    let exp_id = dag.add_node(ExposureOp::new(1.0)); // +1 EV (doubles linear energy)
    let cont_id = dag.add_node(ContrastOp::new(0.2));
    let _sat_id = dag.add_node(SaturationOp::new(0.5));

    assert_eq!(dag.node_count(), 3);

    // Initial render
    let rendered = dag.render(&buf);
    let p = rendered.get_pixel(0, 0).unwrap();
    // After +1 EV, 0.5 sRGB is brighter
    assert!(p.r > 0.5);

    // Disable exposure node and render cached
    dag.set_enabled(exp_id, false);
    let rendered_no_exp = dag.render_cached(&buf);
    let p_no_exp = rendered_no_exp.get_pixel(0, 0).unwrap();
    assert!((p_no_exp.r - 0.5).abs() < 0.1);

    // Remove node
    assert!(dag.remove_node(cont_id));
    assert_eq!(dag.node_count(), 2);
    assert!(!dag.remove_node(99999));

    // Update node
    let ids = dag.node_ids();
    assert_eq!(ids.len(), 2);
    assert!(dag.update_node(exp_id, ExposureOp::new(2.0)));
    assert_eq!(dag.get_node(exp_id).unwrap().name, "Exposure");
    assert!(!dag.update_node(88888, ExposureOp::new(1.0)));
}

#[test]
fn test_buffer_empty_and_bounds() {
    let empty_u8 = PixelBufferRgba8::new(0, 0);
    assert_eq!(empty_u8.pixel_count(), 0);
    assert!(empty_u8.get_pixel(0, 0).is_none());

    let empty_f32 = PixelBufferF32::new(0, 0);
    assert_eq!(empty_f32.pixel_count(), 0);
    assert!(empty_f32.get_pixel(0, 0).is_none());

    // From raw mismatch
    assert!(PixelBufferRgba8::from_raw(2, 2, vec![0u8; 10]).is_err());
    assert!(PixelBufferF32::from_raw(2, 2, vec![0.0f32; 10]).is_err());
}
