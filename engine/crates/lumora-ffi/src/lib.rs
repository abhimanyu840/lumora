//! C-compatible and safe Rust FFI bindings for the Lumora image processing engine.
//!
//! Exposes entry points for mobile host environments (Exact runtime, Android JNI, iOS C-FFI)
//! to load image buffers, manipulate non-destructive develop parameters (exposure, contrast,
//! spline curves, 8-channel HSL, 3-way color grading wheels), add Snapseed-style U-Point pins,
//! execute local AI model enhancements, and render 1080p preview buffers to raw RGBA pointers.

#![allow(clippy::missing_safety_doc)]

use std::ffi::CStr;
use std::os::raw::c_char;
use std::ptr;
use std::slice;
use std::sync::Arc;

use lumora_ai::pipeline::{AiDenoiseOp, AiDepthBokehOp, AiSuperResOp, AiZeroDceOp};
use lumora_core::buffer::{PixelBufferF32, PixelBufferRgba8, RgbaF32};
use lumora_core::dag::{ContrastOp, ExposureOp, ImageOp, PipelineDAG, SaturationOp};
use lumora_core::exporter::{ExportError, ExportFormat, ImageExporter};
use lumora_ops::color_grading::{ColorGrading, GradingWheel};
use lumora_ops::curves::{CurvePoint, ToneCurves};
use lumora_ops::hsl::{ChannelAdjust, HslMixer};
use lumora_ops::presets::LumoraPreset;
use lumora_ops::upoint::{UPointMask, UPointPin};

pub const LUMORA_SUCCESS: i32 = 0;
pub const LUMORA_ERR_NULL_PTR: i32 = -1;
pub const LUMORA_ERR_INVALID_PARAM: i32 = -2;
pub const LUMORA_ERR_PIPELINE: i32 = -3;

/// Core engine session managing image buffers and the non-destructive edit tree.
#[derive(Clone)]
pub struct LumoraImage {
    pub original: Arc<PixelBufferF32>,
    pub preview: PixelBufferRgba8,
    pub exposure: f32,
    pub contrast: f32,
    pub saturation: f32,
    pub curves: ToneCurves,
    pub hsl: HslMixer,
    pub color_grading: ColorGrading,
    pub upoints: UPointMask,
    pub ai_passes: Vec<(i32, f32)>, // (op_type, strength)
}

impl LumoraImage {
    /// Create a new LumoraImage session from RGBA8 pixel data.
    pub fn new(width: u32, height: u32, rgba_data: &[u8]) -> Result<Self, i32> {
        let expected_len = match (width as usize)
            .checked_mul(height as usize)
            .and_then(|wh| wh.checked_mul(4))
        {
            Some(len) => len,
            None => return Err(LUMORA_ERR_INVALID_PARAM),
        };
        if width == 0 || height == 0 || rgba_data.len() != expected_len {
            return Err(LUMORA_ERR_INVALID_PARAM);
        }
        let rgba_buf =
            PixelBufferRgba8::from_raw(width as usize, height as usize, rgba_data.to_vec())
                .map_err(|_| LUMORA_ERR_INVALID_PARAM)?;
        let f32_buf = rgba_buf.to_f32();
        let preview = rgba_buf;

        Ok(Self {
            original: Arc::new(f32_buf),
            preview,
            exposure: 0.0,
            contrast: 0.0,
            saturation: 0.0,
            curves: ToneCurves::linear(),
            hsl: HslMixer::default(),
            color_grading: ColorGrading::default(),
            upoints: UPointMask::new(),
            ai_passes: Vec::new(),
        })
    }

    /// Renders the complete non-destructive pipeline into the preview buffer.
    pub fn render_preview(&mut self, max_width: u32, max_height: u32) -> Result<(), i32> {
        if self.original.width == 0 || self.original.height == 0 {
            return Err(LUMORA_ERR_PIPELINE);
        }

        // Determine working resolution (downsample to fit within max_width x max_height if requested)
        let mut working = if max_width > 0 && max_height > 0 {
            let scale_w = max_width as f32 / self.original.width as f32;
            let scale_h = max_height as f32 / self.original.height as f32;
            let scale = scale_w.min(scale_h).min(1.0);
            let target_w = ((self.original.width as f32 * scale).round() as usize).max(1);
            let target_h = ((self.original.height as f32 * scale).round() as usize).max(1);
            if target_w < self.original.width || target_h < self.original.height {
                self.downsample_bilinear(&self.original, target_w, target_h)
            } else {
                (*self.original).clone()
            }
        } else {
            (*self.original).clone()
        };

        // 1. Exposure
        if self.exposure.abs() > 1e-4 {
            ExposureOp { ev: self.exposure }.apply(&mut working);
        }

        // 2. Contrast
        if self.contrast.abs() > 1e-4 {
            ContrastOp {
                amount: self.contrast,
            }
            .apply(&mut working);
        }

        // 3. Saturation
        if self.saturation.abs() > 1e-4 {
            SaturationOp {
                amount: self.saturation,
            }
            .apply(&mut working);
        }

        // 4. Tone Curves
        if !self.curves.is_identity() {
            self.curves.apply_to_buffer(&mut working);
        }

        // 5. 8-Channel HSL Mixer
        self.hsl.apply_to_buffer(&mut working);

        // 6. 3-Way Color Grading
        self.color_grading.apply_to_buffer(&mut working);

        // 7. U-Point Selective Control Pins
        self.upoints.apply_to_buffer(&mut working);

        // 8. AI Enhancement passes
        for &(op_type, strength) in &self.ai_passes {
            match op_type {
                1 => AiDenoiseOp::new(strength).apply(&mut working),
                2 => AiZeroDceOp::new(strength).apply(&mut working),
                3 => AiSuperResOp::new().apply(&mut working),
                4 => AiDepthBokehOp::new(0.5, strength, 8.0).apply(&mut working),
                _ => {}
            }
        }

        // Convert working float buffer to preview RGBA8
        self.preview = working.to_rgba8();
        Ok(())
    }

    fn downsample_bilinear(
        &self,
        src: &PixelBufferF32,
        dst_w: usize,
        dst_h: usize,
    ) -> PixelBufferF32 {
        let mut dst = PixelBufferF32::new(dst_w, dst_h);
        let x_ratio = src.width as f32 / dst_w as f32;
        let y_ratio = src.height as f32 / dst_h as f32;

        for y in 0..dst_h {
            let src_y = (y as f32 * y_ratio).clamp(0.0, (src.height - 1) as f32);
            let y_low = src_y.floor() as usize;
            let y_high = (y_low + 1).min(src.height - 1);
            let y_weight = src_y - y_low as f32;

            for x in 0..dst_w {
                let src_x = (x as f32 * x_ratio).clamp(0.0, (src.width - 1) as f32);
                let x_low = src_x.floor() as usize;
                let x_high = (x_low + 1).min(src.width - 1);
                let x_weight = src_x - x_low as f32;

                let c00 = src
                    .get_pixel(x_low, y_low)
                    .unwrap_or(RgbaF32::rgb(0.0, 0.0, 0.0));
                let c10 = src
                    .get_pixel(x_high, y_low)
                    .unwrap_or(RgbaF32::rgb(0.0, 0.0, 0.0));
                let c01 = src
                    .get_pixel(x_low, y_high)
                    .unwrap_or(RgbaF32::rgb(0.0, 0.0, 0.0));
                let c11 = src
                    .get_pixel(x_high, y_high)
                    .unwrap_or(RgbaF32::rgb(0.0, 0.0, 0.0));

                let interp = |v00: f32, v10: f32, v01: f32, v11: f32| -> f32 {
                    let top = v00 * (1.0 - x_weight) + v10 * x_weight;
                    let bottom = v01 * (1.0 - x_weight) + v11 * x_weight;
                    top * (1.0 - y_weight) + bottom * y_weight
                };

                let r = interp(c00.r, c10.r, c01.r, c11.r);
                let g = interp(c00.g, c10.g, c01.g, c11.g);
                let b = interp(c00.b, c10.b, c01.b, c11.b);
                let a = interp(c00.a, c10.a, c01.a, c11.a);

                let _ = dst.set_pixel(x, y, RgbaF32::new(r, g, b, a));
            }
        }
        dst
    }

    /// Render and export the original full-resolution image using memory-efficient tiled processing.
    pub fn export_full_resolution(
        &self,
        format: ExportFormat,
        tile_size: usize,
    ) -> Result<Vec<u8>, ExportError> {
        if self.original.width == 0 || self.original.height == 0 {
            return Err(ExportError::EmptyBuffer);
        }

        let tile_sz = if tile_size > 0 { tile_size } else { 512 };
        let exporter = ImageExporter::new(format).with_tile_size(tile_sz);

        let mut dag = PipelineDAG::new();

        // 1. Exposure
        if self.exposure.abs() > 1e-4 {
            dag.add_node(ExposureOp { ev: self.exposure });
        }

        // 2. Contrast
        if self.contrast.abs() > 1e-4 {
            dag.add_node(ContrastOp {
                amount: self.contrast,
            });
        }

        // 3. Saturation
        if self.saturation.abs() > 1e-4 {
            dag.add_node(SaturationOp {
                amount: self.saturation,
            });
        }

        // 4. Tone Curves
        if !self.curves.is_identity() {
            dag.add_node(self.curves.clone());
        }

        // 5. 8-Channel HSL Mixer
        dag.add_node(self.hsl.clone());

        // 6. 3-Way Color Grading
        dag.add_node(self.color_grading.clone());

        // 7. U-Point Selective Control Pins
        if !self.upoints.pins.is_empty() {
            let mut upoints = self.upoints.clone();
            upoints.resolve_targets(&self.original);
            dag.add_node(upoints);
        }

        // 8. AI Enhancement passes
        let mut has_super_res = false;
        for &(op_type, strength) in &self.ai_passes {
            match op_type {
                1 => {
                    dag.add_node(AiDenoiseOp::new(strength));
                }
                2 => {
                    dag.add_node(AiZeroDceOp::new(strength));
                }
                3 => {
                    has_super_res = true;
                }
                4 => {
                    dag.add_node(AiDepthBokehOp::new(0.5, strength, 8.0));
                }
                _ => {}
            }
        }

        if has_super_res {
            let base_rgba = exporter.render_tiled_dag(&self.original, &dag)?;
            let mut base_f32 = base_rgba.to_f32();
            AiSuperResOp::new().apply(&mut base_f32);
            exporter.export_f32(&base_f32)
        } else {
            exporter.export_tiled_dag(&self.original, &dag)
        }
    }
}

// ============================================================================
// C-Compatible Foreign Function Interface (FFI)
// ============================================================================

/// Allocate and initialize a new Lumora image session.
#[no_mangle]
pub unsafe extern "C" fn lumora_image_create(
    width: u32,
    height: u32,
    rgba_data: *const u8,
) -> *mut LumoraImage {
    if rgba_data.is_null() || width == 0 || height == 0 {
        return ptr::null_mut();
    }
    let data_len = match (width as usize)
        .checked_mul(height as usize)
        .and_then(|wh| wh.checked_mul(4))
    {
        Some(len) => len,
        None => return ptr::null_mut(),
    };
    let slice = slice::from_raw_parts(rgba_data, data_len);

    match LumoraImage::new(width, height, slice) {
        Ok(img) => Box::into_raw(Box::new(img)),
        Err(_) => ptr::null_mut(),
    }
}

/// Free a Lumora image session allocated by `lumora_image_create`.
#[no_mangle]
pub unsafe extern "C" fn lumora_image_destroy(image: *mut LumoraImage) {
    if !image.is_null() {
        drop(Box::from_raw(image));
    }
}

/// Retrieve the original full-resolution image width in pixels.
#[no_mangle]
pub unsafe extern "C" fn lumora_image_get_original_width(image: *const LumoraImage) -> u32 {
    if image.is_null() {
        return 0;
    }
    let img = &*image;
    img.original.width as u32
}

/// Retrieve the original full-resolution image height in pixels.
#[no_mangle]
pub unsafe extern "C" fn lumora_image_get_original_height(image: *const LumoraImage) -> u32 {
    if image.is_null() {
        return 0;
    }
    let img = &*image;
    img.original.height as u32
}

/// Retrieve the preview width in pixels.
#[no_mangle]
pub unsafe extern "C" fn lumora_image_get_width(image: *const LumoraImage) -> u32 {
    if image.is_null() {
        return 0;
    }
    (*image).preview.width as u32
}

/// Retrieve the preview height in pixels.
#[no_mangle]
pub unsafe extern "C" fn lumora_image_get_height(image: *const LumoraImage) -> u32 {
    if image.is_null() {
        return 0;
    }
    (*image).preview.height as u32
}

/// Retrieve pointer to the rendered preview RGBA8 byte buffer.
#[no_mangle]
pub unsafe extern "C" fn lumora_image_get_data(image: *const LumoraImage) -> *const u8 {
    if image.is_null() {
        return ptr::null();
    }
    (*image).preview.data.as_ptr()
}

/// Retrieve the byte length of the rendered preview RGBA8 buffer.
#[no_mangle]
pub unsafe extern "C" fn lumora_image_get_data_len(image: *const LumoraImage) -> usize {
    if image.is_null() {
        return 0;
    }
    (*image).preview.data.len()
}

/// Apply exposure adjustment in EV (-5.0 to +5.0).
#[no_mangle]
pub unsafe extern "C" fn lumora_apply_exposure(image: *mut LumoraImage, ev: f32) -> i32 {
    if image.is_null() {
        return LUMORA_ERR_NULL_PTR;
    }
    (*image).exposure = ev.clamp(-5.0, 5.0);
    LUMORA_SUCCESS
}

/// Apply contrast adjustment factor (-1.0 to 1.0).
#[no_mangle]
pub unsafe extern "C" fn lumora_apply_contrast(image: *mut LumoraImage, factor: f32) -> i32 {
    if image.is_null() {
        return LUMORA_ERR_NULL_PTR;
    }
    (*image).contrast = factor.clamp(-1.0, 1.0);
    LUMORA_SUCCESS
}

/// Apply saturation adjustment factor (-1.0 to 1.0).
#[no_mangle]
pub unsafe extern "C" fn lumora_apply_saturation(image: *mut LumoraImage, factor: f32) -> i32 {
    if image.is_null() {
        return LUMORA_ERR_NULL_PTR;
    }
    (*image).saturation = factor.clamp(-1.0, 1.0);
    LUMORA_SUCCESS
}

/// Apply parametric spline curves from pairs of [x, y] coordinates.
#[no_mangle]
pub unsafe extern "C" fn lumora_apply_curves(
    image: *mut LumoraImage,
    points_ptr: *const f32,
    point_count: usize,
) -> i32 {
    if image.is_null() {
        return LUMORA_ERR_NULL_PTR;
    }
    if points_ptr.is_null() || point_count < 2 {
        return LUMORA_ERR_INVALID_PARAM;
    }

    let coords = slice::from_raw_parts(points_ptr, point_count * 2);
    let mut curve_points = Vec::with_capacity(point_count);
    for i in 0..point_count {
        curve_points.push(CurvePoint::new(coords[i * 2], coords[i * 2 + 1]));
    }
    (*image).curves.master = curve_points;
    LUMORA_SUCCESS
}

/// Apply parametric spline curves to a specific channel.
/// `channel_idx`: 0=Master, 1=Red, 2=Green, 3=Blue.
#[no_mangle]
pub unsafe extern "C" fn lumora_apply_channel_curves(
    image: *mut LumoraImage,
    channel_idx: i32,
    points_ptr: *const f32,
    point_count: usize,
) -> i32 {
    if image.is_null() {
        return LUMORA_ERR_NULL_PTR;
    }
    if points_ptr.is_null() || point_count < 2 {
        return LUMORA_ERR_INVALID_PARAM;
    }
    if !(0..=3).contains(&channel_idx) {
        return LUMORA_ERR_INVALID_PARAM;
    }

    let coords = slice::from_raw_parts(points_ptr, point_count * 2);
    let mut curve_points = Vec::with_capacity(point_count);
    for i in 0..point_count {
        curve_points.push(CurvePoint::new(coords[i * 2], coords[i * 2 + 1]));
    }
    let curves = &mut (*image).curves;
    match channel_idx {
        0 => curves.master = curve_points,
        1 => curves.red = curve_points,
        2 => curves.green = curve_points,
        3 => curves.blue = curve_points,
        _ => {}
    }
    LUMORA_SUCCESS
}

/// Apply 8-channel HSL color shift.
/// `channel_idx`: 0=Red, 1=Orange, 2=Yellow, 3=Green, 4=Aqua, 5=Blue, 6=Purple, 7=Magenta.
#[no_mangle]
pub unsafe extern "C" fn lumora_apply_hsl(
    image: *mut LumoraImage,
    channel_idx: i32,
    hue_shift: f32,
    sat_shift: f32,
    lum_shift: f32,
) -> i32 {
    if image.is_null() {
        return LUMORA_ERR_NULL_PTR;
    }
    if !(0..=7).contains(&channel_idx) {
        return LUMORA_ERR_INVALID_PARAM;
    }

    let adjust = ChannelAdjust {
        hue: hue_shift.clamp(-30.0, 30.0),
        saturation: sat_shift.clamp(-1.0, 1.0),
        luminance: lum_shift.clamp(-1.0, 1.0),
    };

    let mixer = &mut (*image).hsl;
    match channel_idx {
        0 => mixer.red = adjust,
        1 => mixer.orange = adjust,
        2 => mixer.yellow = adjust,
        3 => mixer.green = adjust,
        4 => mixer.aqua = adjust,
        5 => mixer.blue = adjust,
        6 => mixer.purple = adjust,
        7 => mixer.magenta = adjust,
        _ => {}
    }

    LUMORA_SUCCESS
}

/// Apply 3-way color grading split toning.
#[no_mangle]
pub unsafe extern "C" fn lumora_apply_color_grading(
    image: *mut LumoraImage,
    shadows_h: f32,
    shadows_s: f32,
    highlights_h: f32,
    highlights_s: f32,
) -> i32 {
    if image.is_null() {
        return LUMORA_ERR_NULL_PTR;
    }

    let grading = &mut (*image).color_grading;
    grading.shadows = GradingWheel {
        hue: shadows_h.clamp(0.0, 360.0),
        saturation: shadows_s.clamp(0.0, 1.0),
        luminance: 0.0,
    };
    grading.highlights = GradingWheel {
        hue: highlights_h.clamp(0.0, 360.0),
        saturation: highlights_s.clamp(0.0, 1.0),
        luminance: 0.0,
    };

    LUMORA_SUCCESS
}

/// Apply full 3-way color grading split toning with Shadows, Midtones, Highlights, Balance, and Blending.
#[no_mangle]
pub unsafe extern "C" fn lumora_apply_color_grading_full(
    image: *mut LumoraImage,
    shadows_h: f32,
    shadows_s: f32,
    shadows_l: f32,
    midtones_h: f32,
    midtones_s: f32,
    midtones_l: f32,
    highlights_h: f32,
    highlights_s: f32,
    highlights_l: f32,
    balance: f32,
    blending: f32,
) -> i32 {
    if image.is_null() {
        return LUMORA_ERR_NULL_PTR;
    }

    let grading = &mut (*image).color_grading;
    grading.shadows = GradingWheel {
        hue: shadows_h.clamp(0.0, 360.0),
        saturation: shadows_s.clamp(0.0, 1.0),
        luminance: shadows_l.clamp(-1.0, 1.0),
    };
    grading.midtones = GradingWheel {
        hue: midtones_h.clamp(0.0, 360.0),
        saturation: midtones_s.clamp(0.0, 1.0),
        luminance: midtones_l.clamp(-1.0, 1.0),
    };
    grading.highlights = GradingWheel {
        hue: highlights_h.clamp(0.0, 360.0),
        saturation: highlights_s.clamp(0.0, 1.0),
        luminance: highlights_l.clamp(-1.0, 1.0),
    };
    grading.balance = balance.clamp(-1.0, 1.0);
    grading.blending = blending.clamp(0.0, 1.0);

    LUMORA_SUCCESS
}

/// Add a Snapseed-style U-Point selective control pin.
#[no_mangle]
pub unsafe extern "C" fn lumora_add_upoint(
    image: *mut LumoraImage,
    x: f32,
    y: f32,
    radius: f32,
    brightness: f32,
    contrast: f32,
    saturation: f32,
) -> i32 {
    if image.is_null() {
        return LUMORA_ERR_NULL_PTR;
    }

    let img = &mut *image;
    let mut pin = UPointPin::new(x, y, radius);
    pin.brightness = brightness.clamp(-1.0, 1.0);
    pin.contrast = contrast.clamp(-1.0, 1.0);
    pin.saturation = saturation.clamp(-1.0, 1.0);

    let width = img.original.width;
    let height = img.original.height;
    if width > 0 && height > 0 {
        let px = ((pin.center_x * (width - 1) as f32) as usize).min(width.saturating_sub(1));
        let py = ((pin.center_y * (height - 1) as f32) as usize).min(height.saturating_sub(1));
        if let Some(p) = img.original.get_pixel(px, py) {
            pin.target_lab = Some(lumora_core::color::srgb_to_lab(p.r, p.g, p.b));
        }
    }

    img.upoints.add_pin(pin);
    LUMORA_SUCCESS
}

/// Clear all active U-Point pins.
#[no_mangle]
pub unsafe extern "C" fn lumora_clear_upoints(image: *mut LumoraImage) -> i32 {
    if image.is_null() {
        return LUMORA_ERR_NULL_PTR;
    }
    (*image).upoints.pins.clear();
    LUMORA_SUCCESS
}

/// Trigger an on-device AI enhancement pass.
/// `op_type`: 1 = DnCNN Denoise, 2 = Zero-DCE Low-Light, 3 = Real-ESRGAN 4x Upscale, 4 = MiDaS Depth Bokeh.
#[no_mangle]
pub unsafe extern "C" fn lumora_apply_ai_enhancement(
    image: *mut LumoraImage,
    op_type: i32,
    strength: f32,
) -> i32 {
    if image.is_null() {
        return LUMORA_ERR_NULL_PTR;
    }
    if !(1..=4).contains(&op_type) {
        return LUMORA_ERR_INVALID_PARAM;
    }

    let clamped_strength = strength.clamp(0.0, 1.0);
    let passes = &mut (*image).ai_passes;
    if let Some(existing) = passes.iter_mut().find(|(t, _)| *t == op_type) {
        existing.1 = clamped_strength;
    } else {
        passes.push((op_type, clamped_strength));
    }
    LUMORA_SUCCESS
}

/// Clear all scheduled AI enhancement passes.
#[no_mangle]
pub unsafe extern "C" fn lumora_clear_ai_enhancements(image: *mut LumoraImage) -> i32 {
    if image.is_null() {
        return LUMORA_ERR_NULL_PTR;
    }
    (*image).ai_passes.clear();
    LUMORA_SUCCESS
}

/// Render the complete edit pipeline into the internal preview buffer.
#[no_mangle]
pub unsafe extern "C" fn lumora_render_preview(
    image: *mut LumoraImage,
    out_rgba: *mut u8,
    max_width: u32,
    max_height: u32,
) -> i32 {
    if image.is_null() {
        return LUMORA_ERR_NULL_PTR;
    }

    let res = (*image).render_preview(max_width, max_height);
    if res.is_err() {
        return LUMORA_ERR_PIPELINE;
    }

    // If an external output buffer pointer was provided, copy the rendered bytes into it
    if !out_rgba.is_null() {
        let preview_len = (*image).preview.data.len();
        ptr::copy_nonoverlapping((*image).preview.data.as_ptr(), out_rgba, preview_len);
    }

    LUMORA_SUCCESS
}

/// Render the complete edit pipeline with explicit output buffer capacity verification.
#[no_mangle]
pub unsafe extern "C" fn lumora_render_preview_sized(
    image: *mut LumoraImage,
    out_rgba: *mut u8,
    out_len: usize,
    max_width: u32,
    max_height: u32,
) -> i32 {
    if image.is_null() {
        return LUMORA_ERR_NULL_PTR;
    }

    let res = (*image).render_preview(max_width, max_height);
    if res.is_err() {
        return LUMORA_ERR_PIPELINE;
    }

    if !out_rgba.is_null() {
        let preview_len = (*image).preview.data.len();
        if out_len < preview_len {
            return LUMORA_ERR_INVALID_PARAM;
        }
        ptr::copy_nonoverlapping((*image).preview.data.as_ptr(), out_rgba, preview_len);
    }

    LUMORA_SUCCESS
}

/// Parse and apply a `.lumora` JSON preset to the current image session.
#[no_mangle]
pub unsafe extern "C" fn lumora_apply_preset_json(
    image: *mut LumoraImage,
    json_c_str: *const c_char,
) -> i32 {
    if image.is_null() || json_c_str.is_null() {
        return LUMORA_ERR_NULL_PTR;
    }

    let c_str = CStr::from_ptr(json_c_str);
    let str_slice = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return LUMORA_ERR_INVALID_PARAM,
    };

    match LumoraPreset::from_json(str_slice) {
        Ok(preset) => {
            let img = &mut (*image);
            if let Some(exp) = preset.exposure {
                img.exposure = exp;
            }
            if let Some(cont) = preset.contrast {
                img.contrast = cont;
            }
            if let Some(sat) = preset.saturation {
                img.saturation = sat;
            }
            if let Some(curves) = preset.curves {
                img.curves = curves;
            }
            if let Some(hsl) = preset.hsl {
                img.hsl = hsl;
            }
            if let Some(grading) = preset.color_grading {
                img.color_grading = grading;
            }
            LUMORA_SUCCESS
        }
        Err(_) => LUMORA_ERR_INVALID_PARAM,
    }
}

/// Export the full-resolution image using memory-efficient tiled processing.
///
/// * `image`: Pointer to LumoraImage session.
/// * `format_code`: 0 = JPEG, 1 = PNG, 2 = WebP.
/// * `quality`: Quality parameter (1..=100 for JPEG; 0 = default 90).
/// * `tile_size`: Tile dimension in pixels (e.g. 512, 1024; 0 = default 512).
/// * `out_data`: Pointer to receive the allocated output byte buffer.
/// * `out_len`: Pointer to receive the length of the output buffer.
///
/// The caller must free `*out_data` using `lumora_free_export_data`.
#[no_mangle]
pub unsafe extern "C" fn lumora_export_image(
    image: *const LumoraImage,
    format_code: i32,
    quality: i32,
    tile_size: u32,
    out_data: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    if image.is_null() || out_data.is_null() || out_len.is_null() {
        return LUMORA_ERR_NULL_PTR;
    }

    // Defensively initialize output pointers
    *out_data = ptr::null_mut();
    *out_len = 0;

    let q = if (1..=100).contains(&quality) {
        quality as u8
    } else {
        90
    };

    let format = match format_code {
        0 => ExportFormat::Jpeg { quality: q },
        1 => ExportFormat::Png,
        2 => ExportFormat::WebP { lossless: true },
        _ => return LUMORA_ERR_INVALID_PARAM,
    };

    let tile_sz = if tile_size > 0 {
        tile_size as usize
    } else {
        512
    };

    match (*image).export_full_resolution(format, tile_sz) {
        Ok(bytes) => {
            *out_len = bytes.len();
            let mut boxed = bytes.into_boxed_slice();
            *out_data = boxed.as_mut_ptr();
            std::mem::forget(boxed);
            LUMORA_SUCCESS
        }
        Err(_) => {
            *out_data = ptr::null_mut();
            *out_len = 0;
            LUMORA_ERR_PIPELINE
        }
    }
}

/// Free an export data buffer allocated by `lumora_export_image`.
#[no_mangle]
pub unsafe extern "C" fn lumora_free_export_data(data: *mut u8, len: usize) {
    if !data.is_null() && len > 0 {
        let slice_ptr = ptr::slice_from_raw_parts_mut(data, len);
        drop(Box::from_raw(slice_ptr));
    }
}
