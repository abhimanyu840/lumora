use lumora_mobile_logic::EngineBridgeState;

#[test]
fn test_bridge_image_lifecycle_and_events() {
    let mut bridge = EngineBridgeState::new();

    // 1. Initial state with no image
    let res = bridge.handle_event("update_exposure", r#"{"ev": 0.5}"#);
    assert!(!res.success);
    assert!(res.error.is_some());

    // 2. Load 32x32 image
    let width = 32u32;
    let height = 32u32;
    let raw_rgba = vec![128u8; (width * height * 4) as usize];
    assert!(bridge.load_image(width, height, &raw_rgba).is_ok());

    assert_eq!(bridge.get_preview_dimensions(), Some((32, 32)));
    assert_eq!(
        bridge.get_preview_data().unwrap().len(),
        (32 * 32 * 4) as usize
    );

    // 3. Exposure event
    let res_exp = bridge.handle_event("update_exposure", r#"{"ev": 0.75}"#);
    assert!(res_exp.success);
    assert_eq!(res_exp.event, "update_exposure");

    // 4. Contrast & Saturation events
    let res_cont = bridge.handle_event("update_contrast", r#"{"factor": 0.2}"#);
    assert!(res_cont.success);

    let res_sat = bridge.handle_event("update_saturation", r#"{"factor": -0.3}"#);
    assert!(res_sat.success);

    // 5. Spline Curves event
    let curves_payload = r#"{
        "channel": "master",
        "points": [[0.0, 0.0], [0.5, 0.6], [1.0, 1.0]]
    }"#;
    let res_curves = bridge.handle_event("curve_point_updated", curves_payload);
    assert!(res_curves.success);

    // 6. HSL Mixer event
    let hsl_payload = r#"{
        "channel": 0,
        "hue": 10.0,
        "sat": 0.25,
        "lum": -0.1
    }"#;
    let res_hsl = bridge.handle_event("update_hsl", hsl_payload);
    assert!(res_hsl.success);

    // 7. U-Point Pin event
    let upoint_payload = r#"{
        "x": 0.5,
        "y": 0.5,
        "radius": 0.3,
        "brightness": 0.2,
        "contrast": 0.1,
        "saturation": 0.0
    }"#;
    let res_upoint = bridge.handle_event("add_new_upoint_pin", upoint_payload);
    assert!(res_upoint.success);

    // 8. AI pass event
    let ai_payload = r#"{"op_type": 1, "strength": 0.8}"#;
    let res_ai = bridge.handle_event("apply_ai_pass", ai_payload);
    assert!(res_ai.success);

    // 9. Undo & Redo verification
    assert!(bridge.undo());
    assert!(bridge.redo());

    // 10. Unknown event handling
    let res_unknown = bridge.handle_event("unknown_bogus_event", "{}");
    assert!(!res_unknown.success);
    assert!(res_unknown.error.unwrap().contains("Unknown event"));
}

#[test]
fn test_bridge_contract_specific_ui_events() {
    let mut bridge = EngineBridgeState::new();
    let raw_rgba = vec![150u8; 16 * 16 * 4];
    assert!(bridge.load_image(16, 16, &raw_rgba).is_ok());

    // 1. Individual HSL slider dials from develop.contract
    assert!(
        bridge
            .handle_event("update_hsl_hue", r#"{"channel": 1, "val": -15.0}"#)
            .success
    );
    assert!(
        bridge
            .handle_event("update_hsl_sat", r#"{"channel": 1, "val": 0.4}"#)
            .success
    );
    assert!(
        bridge
            .handle_event("update_hsl_lum", r#"{"channel": 1, "val": 0.1}"#)
            .success
    );

    // 2. Color grading dials from develop.contract
    assert!(
        bridge
            .handle_event("update_shadows_hue", r#"{"val": 220.0}"#)
            .success
    );
    assert!(
        bridge
            .handle_event("update_highlights_hue", r#"{"val": 40.0}"#)
            .success
    );
    assert!(bridge.handle_event("update_color_grading", r#"{"shadows_h": 200.0, "shadows_s": 0.3, "highlights_h": 50.0, "highlights_s": 0.4}"#).success);

    // 3. Single curve coordinate from develop.contract canvas
    assert!(
        bridge
            .handle_event(
                "curve_point_updated",
                r#"{"channel": "red", "x": 0.5, "y": 0.7}"#
            )
            .success
    );

    // 4. Default U-Point pin from selective.contract ("+ Add Pin" with empty payload)
    assert!(bridge.handle_event("add_new_upoint_pin", "{}").success);
    // Pin slider adjustments from selective.contract
    assert!(
        bridge
            .handle_event("update_pin_radius", r#"{"index": 0, "radius": 0.45}"#)
            .success
    );
    assert!(
        bridge
            .handle_event("update_pin_brightness", r#"{"index": 0, "val": 0.3}"#)
            .success
    );
    assert!(
        bridge
            .handle_event("update_pin_contrast", r#"{"index": 0, "val": -0.2}"#)
            .success
    );
    assert!(
        bridge
            .handle_event("update_pin_saturation", r#"{"index": 0, "val": 0.15}"#)
            .success
    );

    // 5. AI passes using "type" alias from ai_lab.contract
    assert!(
        bridge
            .handle_event("apply_ai_pass", r#"{"type": 1, "strength": 0.6}"#)
            .success
    );
    assert!(
        bridge
            .handle_event("apply_ai_pass", r#"{"type": 2, "strength": 0.5}"#)
            .success
    );
    assert!(
        bridge
            .handle_event("apply_ai_pass", r#"{"type": 4, "strength": 0.8}"#)
            .success
    );

    // 6. Scene auto enhancement from ai_lab.contract
    assert!(
        bridge
            .handle_event("apply_scene_auto_enhance", r#"{"scene": "Portrait"}"#)
            .success
    );
    assert!(
        bridge
            .handle_event("apply_scene_auto_enhance", r#"{"scene": "Landscape"}"#)
            .success
    );

    // 7. Preset selection from presets.contract
    assert!(
        bridge
            .handle_event("apply_preset_to_engine", r#"{"preset_id": 1}"#)
            .success
    );
    assert!(
        bridge
            .handle_event("apply_preset_to_engine", r#"{"preset_id": 2}"#)
            .success
    );

    // 8. Direct preset JSON
    let raw_preset_json = r#"{"name": "Warm Glow", "exposure": 0.2, "saturation": 0.1}"#;
    assert!(
        bridge
            .handle_event("apply_preset_json", raw_preset_json)
            .success
    );

    // 9. Full export
    assert!(bridge.handle_event("export_rendered_image", "{}").success);

    // 10. Clear pins
    assert!(bridge.handle_event("clear_all_upoint_pins", "{}").success);
}
