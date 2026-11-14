use std::ffi::CString;
use std::ptr;

use lumora_ffi::*;
use lumora_ops::presets::LumoraPreset;

#[test]
fn test_ffi_image_lifecycle_and_dimensions() {
    let width = 64u32;
    let height = 64u32;
    let raw_rgba = vec![128u8; (width * height * 4) as usize];

    unsafe {
        // 1. Create image
        let handle = lumora_image_create(width, height, raw_rgba.as_ptr());
        assert!(!handle.is_null());

        // 2. Query dimensions
        assert_eq!(lumora_image_get_width(handle), width);
        assert_eq!(lumora_image_get_height(handle), height);
        assert_eq!(
            lumora_image_get_data_len(handle),
            (width * height * 4) as usize
        );

        let data_ptr = lumora_image_get_data(handle);
        assert!(!data_ptr.is_null());

        // 3. Destroy image
        lumora_image_destroy(handle);
    }
}

#[test]
fn test_ffi_null_safety_and_invalid_params() {
    unsafe {
        // Null image pointer handling
        assert_eq!(
            lumora_apply_exposure(ptr::null_mut(), 1.0),
            LUMORA_ERR_NULL_PTR
        );
        assert_eq!(
            lumora_apply_contrast(ptr::null_mut(), 0.5),
            LUMORA_ERR_NULL_PTR
        );
        assert_eq!(
            lumora_apply_saturation(ptr::null_mut(), 0.5),
            LUMORA_ERR_NULL_PTR
        );
        assert_eq!(
            lumora_apply_curves(ptr::null_mut(), ptr::null(), 0),
            LUMORA_ERR_NULL_PTR
        );
        assert_eq!(
            lumora_apply_hsl(ptr::null_mut(), 0, 10.0, 0.2, 0.1),
            LUMORA_ERR_NULL_PTR
        );
        assert_eq!(
            lumora_apply_color_grading(ptr::null_mut(), 30.0, 0.5, 210.0, 0.5),
            LUMORA_ERR_NULL_PTR
        );
        assert_eq!(
            lumora_add_upoint(ptr::null_mut(), 0.5, 0.5, 0.2, 0.1, 0.0, 0.0),
            LUMORA_ERR_NULL_PTR
        );
        assert_eq!(lumora_clear_upoints(ptr::null_mut()), LUMORA_ERR_NULL_PTR);
        assert_eq!(
            lumora_apply_ai_enhancement(ptr::null_mut(), 1, 0.8),
            LUMORA_ERR_NULL_PTR
        );
        assert_eq!(
            lumora_clear_ai_enhancements(ptr::null_mut()),
            LUMORA_ERR_NULL_PTR
        );
        assert_eq!(
            lumora_render_preview(ptr::null_mut(), ptr::null_mut(), 1080, 1080),
            LUMORA_ERR_NULL_PTR
        );
        assert_eq!(
            lumora_apply_preset_json(ptr::null_mut(), ptr::null()),
            LUMORA_ERR_NULL_PTR
        );

        // Invalid creation parameters
        let dummy = [255u8; 16];
        assert!(lumora_image_create(0, 0, dummy.as_ptr()).is_null());
        assert!(lumora_image_create(2, 2, ptr::null()).is_null());
    }
}

#[test]
fn test_ffi_develop_adjustments_and_render() {
    let width = 32u32;
    let height = 32u32;
    // Mid-gray image (128, 128, 128, 255)
    let mut raw_rgba = Vec::with_capacity((width * height * 4) as usize);
    for _ in 0..(width * height) {
        raw_rgba.extend_from_slice(&[128, 128, 128, 255]);
    }

    unsafe {
        let handle = lumora_image_create(width, height, raw_rgba.as_ptr());
        assert!(!handle.is_null());

        // 1. Exposure shift (+1.0 EV)
        assert_eq!(lumora_apply_exposure(handle, 1.0), LUMORA_SUCCESS);
        assert_eq!(
            lumora_render_preview(handle, ptr::null_mut(), 0, 0),
            LUMORA_SUCCESS
        );

        let data_ptr = lumora_image_get_data(handle);
        let preview_slice = std::slice::from_raw_parts(data_ptr, (width * height * 4) as usize);
        // Mid-gray 128 under +1 EV should be brighter than 128
        assert!(preview_slice[0] > 128);

        // 2. Contrast & Saturation
        assert_eq!(lumora_apply_contrast(handle, 0.2), LUMORA_SUCCESS);
        assert_eq!(lumora_apply_saturation(handle, -0.5), LUMORA_SUCCESS);

        // 3. Spline curves
        let curve_points: [f32; 8] = [0.0, 0.0, 0.25, 0.2, 0.75, 0.8, 1.0, 1.0];
        assert_eq!(
            lumora_apply_curves(handle, curve_points.as_ptr(), 4),
            LUMORA_SUCCESS
        );

        // 4. HSL adjustment
        assert_eq!(lumora_apply_hsl(handle, 0, 15.0, 0.4, 0.1), LUMORA_SUCCESS); // Red channel
        assert_eq!(lumora_apply_hsl(handle, 3, -10.0, 0.2, 0.0), LUMORA_SUCCESS); // Green channel
        assert_eq!(
            lumora_apply_hsl(handle, 99, 0.0, 0.0, 0.0),
            LUMORA_ERR_INVALID_PARAM
        ); // Invalid channel

        // 5. Color grading
        assert_eq!(
            lumora_apply_color_grading(handle, 220.0, 0.3, 40.0, 0.4),
            LUMORA_SUCCESS
        );

        // 6. U-Point selective pin
        assert_eq!(
            lumora_add_upoint(handle, 0.5, 0.5, 0.3, 0.2, 0.1, 0.1),
            LUMORA_SUCCESS
        );

        // 7. Render with preview downsampling to 16x16
        let mut out_buffer = vec![0u8; 16 * 16 * 4];
        assert_eq!(
            lumora_render_preview(handle, out_buffer.as_mut_ptr(), 16, 16),
            LUMORA_SUCCESS
        );
        assert_eq!(lumora_image_get_width(handle), 16);
        assert_eq!(lumora_image_get_height(handle), 16);
        assert_eq!(out_buffer[3], 255); // Alpha preserved

        // 8. Clear U-points
        assert_eq!(lumora_clear_upoints(handle), LUMORA_SUCCESS);

        lumora_image_destroy(handle);
    }
}

#[test]
fn test_ffi_ai_passes_and_preset_json() {
    let width = 16u32;
    let height = 16u32;
    let raw_rgba = vec![100u8; (width * height * 4) as usize];

    unsafe {
        let handle = lumora_image_create(width, height, raw_rgba.as_ptr());
        assert!(!handle.is_null());

        // 1. AI Enhancement passes
        assert_eq!(lumora_apply_ai_enhancement(handle, 1, 0.75), LUMORA_SUCCESS); // Denoise
        assert_eq!(lumora_apply_ai_enhancement(handle, 2, 0.5), LUMORA_SUCCESS); // Zero-DCE
        assert_eq!(lumora_apply_ai_enhancement(handle, 3, 1.0), LUMORA_SUCCESS); // SuperRes
        assert_eq!(lumora_apply_ai_enhancement(handle, 4, 0.8), LUMORA_SUCCESS); // Bokeh
        assert_eq!(
            lumora_apply_ai_enhancement(handle, 9, 0.5),
            LUMORA_ERR_INVALID_PARAM
        ); // Invalid

        assert_eq!(
            lumora_render_preview(handle, ptr::null_mut(), 0, 0),
            LUMORA_SUCCESS
        );
        assert_eq!(lumora_clear_ai_enhancements(handle), LUMORA_SUCCESS);

        // 2. Preset JSON application using LumoraPreset
        let mut preset = LumoraPreset::new("Moody Film", "Lumora");
        preset.exposure = Some(0.25);
        preset.contrast = Some(0.15);
        preset.saturation = Some(0.10);
        let preset_json = preset.to_json().unwrap();

        let c_str = CString::new(preset_json).unwrap();
        assert_eq!(
            lumora_apply_preset_json(handle, c_str.as_ptr()),
            LUMORA_SUCCESS
        );

        assert_eq!(
            lumora_render_preview(handle, ptr::null_mut(), 0, 0),
            LUMORA_SUCCESS
        );

        lumora_image_destroy(handle);
    }
}

#[test]
fn test_ffi_dimensions_and_sized_render() {
    let width = 64u32;
    let height = 32u32;
    let raw_rgba = vec![200u8; (width * height * 4) as usize];

    unsafe {
        let handle = lumora_image_create(width, height, raw_rgba.as_ptr());
        assert!(!handle.is_null());

        // Check original vs preview dimensions
        assert_eq!(lumora_image_get_original_width(handle), 64);
        assert_eq!(lumora_image_get_original_height(handle), 32);

        // Render preview downsampled to 16x16 (which preserves aspect ratio: 16x8)
        let mut small_buf = vec![0u8; 16 * 8 * 4];
        let res =
            lumora_render_preview_sized(handle, small_buf.as_mut_ptr(), small_buf.len(), 16, 16);
        assert_eq!(res, LUMORA_SUCCESS);
        assert_eq!(lumora_image_get_width(handle), 16);
        assert_eq!(lumora_image_get_height(handle), 8);

        // Passing buffer that is too small returns error
        let mut too_small = vec![0u8; 10];
        let err_res =
            lumora_render_preview_sized(handle, too_small.as_mut_ptr(), too_small.len(), 16, 16);
        assert_eq!(err_res, LUMORA_ERR_INVALID_PARAM);

        lumora_image_destroy(handle);
    }
}

#[test]
fn test_ffi_channel_curves_and_full_grading() {
    let width = 16u32;
    let height = 16u32;
    let raw_rgba = vec![128u8; (width * height * 4) as usize];

    unsafe {
        let handle = lumora_image_create(width, height, raw_rgba.as_ptr());
        assert!(!handle.is_null());

        // Red channel curves
        let pts: [f32; 4] = [0.0, 0.0, 1.0, 0.9];
        assert_eq!(
            lumora_apply_channel_curves(handle, 1, pts.as_ptr(), 2),
            LUMORA_SUCCESS
        );
        // Invalid channel index
        assert_eq!(
            lumora_apply_channel_curves(handle, 5, pts.as_ptr(), 2),
            LUMORA_ERR_INVALID_PARAM
        );

        // Full 3-way color grading
        assert_eq!(
            lumora_apply_color_grading_full(
                handle, 210.0, 0.4, -0.1, // shadows (teal)
                45.0, 0.2, 0.0, // midtones (warm)
                30.0, 0.5, 0.1, // highlights (orange)
                -0.2, 0.6 // balance, blending
            ),
            LUMORA_SUCCESS
        );

        // AI pass deduplication/update
        assert_eq!(lumora_apply_ai_enhancement(handle, 1, 0.3), LUMORA_SUCCESS);
        assert_eq!(lumora_apply_ai_enhancement(handle, 1, 0.7), LUMORA_SUCCESS);
        let img = &*handle;
        // Should only have 1 pass for type 1 with strength 0.7
        let passes: Vec<_> = img.ai_passes.iter().filter(|(t, _)| *t == 1).collect();
        assert_eq!(passes.len(), 1);
        assert_eq!(passes[0].1, 0.7);

        assert_eq!(
            lumora_render_preview(handle, ptr::null_mut(), 0, 0),
            LUMORA_SUCCESS
        );

        lumora_image_destroy(handle);
    }
}

#[test]
fn test_ffi_export_image_all_formats_and_null_safety() {
    let width = 64u32;
    let height = 48u32;
    let mut raw_rgba = Vec::with_capacity((width * height * 4) as usize);
    for i in 0..(width * height) {
        let r = (i % 256) as u8;
        let g = ((i * 3) % 256) as u8;
        let b = ((i * 7) % 256) as u8;
        raw_rgba.extend_from_slice(&[r, g, b, 255]);
    }

    unsafe {
        // Null safety checks
        let mut out_data: *mut u8 = ptr::null_mut();
        let mut out_len: usize = 0;
        assert_eq!(
            lumora_export_image(ptr::null(), 0, 90, 32, &mut out_data, &mut out_len),
            LUMORA_ERR_NULL_PTR
        );

        let handle = lumora_image_create(width, height, raw_rgba.as_ptr());
        assert!(!handle.is_null());

        assert_eq!(
            lumora_export_image(handle, 0, 90, 32, ptr::null_mut(), &mut out_len),
            LUMORA_ERR_NULL_PTR
        );

        assert_eq!(
            lumora_export_image(handle, 0, 90, 32, &mut out_data, ptr::null_mut()),
            LUMORA_ERR_NULL_PTR
        );

        // Invalid format code: verify out pointers are nulled
        out_data = 0x1234 as *mut u8;
        out_len = 9999;
        assert_eq!(
            lumora_export_image(handle, 99, 90, 32, &mut out_data, &mut out_len),
            LUMORA_ERR_INVALID_PARAM
        );
        assert!(out_data.is_null());
        assert_eq!(out_len, 0);

        // 1. Export JPEG (format_code = 0)
        assert_eq!(
            lumora_export_image(handle, 0, 85, 32, &mut out_data, &mut out_len),
            LUMORA_SUCCESS
        );
        assert!(!out_data.is_null());
        assert!(out_len > 0);
        let jpeg_slice = std::slice::from_raw_parts(out_data, out_len);
        assert_eq!(&jpeg_slice[0..3], &[0xFF, 0xD8, 0xFF]);
        lumora_free_export_data(out_data, out_len);

        // 2. Export PNG (format_code = 1)
        out_data = ptr::null_mut();
        out_len = 0;
        assert_eq!(
            lumora_export_image(handle, 1, 0, 32, &mut out_data, &mut out_len),
            LUMORA_SUCCESS
        );
        assert!(!out_data.is_null());
        assert!(out_len > 0);
        let png_slice = std::slice::from_raw_parts(out_data, out_len);
        assert_eq!(
            &png_slice[0..8],
            &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]
        );
        lumora_free_export_data(out_data, out_len);

        // 3. Export WebP (format_code = 2)
        out_data = ptr::null_mut();
        out_len = 0;
        assert_eq!(
            lumora_export_image(handle, 2, 0, 32, &mut out_data, &mut out_len),
            LUMORA_SUCCESS
        );
        assert!(!out_data.is_null());
        assert!(out_len > 0);
        let webp_slice = std::slice::from_raw_parts(out_data, out_len);
        assert_eq!(&webp_slice[0..4], b"RIFF");
        assert_eq!(&webp_slice[8..12], b"WEBP");
        lumora_free_export_data(out_data, out_len);

        // Free null data should be safe no-op
        lumora_free_export_data(ptr::null_mut(), 0);

        lumora_image_destroy(handle);
    }
}
