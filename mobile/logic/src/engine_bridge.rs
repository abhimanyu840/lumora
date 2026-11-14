//! Exact Runtime Logic Module: Engine Bridge.
//!
//! Intercepts Contract declarative UI events (slider scrubbing, U-point pinning,
//! AI enhancements, preset selection) and translates them into transactions on the
//! Rust non-destructive DAG, generating sub-16ms 1080p preview frames.

use lumora_ffi::LumoraImage;
use lumora_ops::color_grading::GradingWheel;
use lumora_ops::curves::CurvePoint;
use lumora_ops::hsl::ChannelAdjust;
use lumora_ops::presets::LumoraPreset;
use lumora_ops::upoint::UPointPin;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct EventResponse {
    pub success: bool,
    pub event: String,
    pub width: u32,
    pub height: u32,
    pub error: Option<String>,
}

#[derive(Default)]
pub struct EngineBridgeState {
    pub current_image: Option<LumoraImage>,
    pub undo_stack: Vec<LumoraImage>,
    pub redo_stack: Vec<LumoraImage>,
}

impl EngineBridgeState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Load an RGBA8 buffer into the editing pipeline.
    pub fn load_image(&mut self, width: u32, height: u32, data: &[u8]) -> Result<(), String> {
        let img = LumoraImage::new(width, height, data)
            .map_err(|e| format!("Failed to create image: error code {e}"))?;
        self.current_image = Some(img);
        self.undo_stack.clear();
        self.redo_stack.clear();
        Ok(())
    }

    fn push_undo(&mut self) {
        if let Some(ref img) = self.current_image {
            // Keep maximum 30 undo steps to balance memory and history depth
            if self.undo_stack.len() >= 30 {
                self.undo_stack.remove(0);
            }
            self.undo_stack.push(img.clone());
            self.redo_stack.clear();
        }
    }

    /// Undo last edit action.
    pub fn undo(&mut self) -> bool {
        if let Some(prev) = self.undo_stack.pop() {
            if let Some(current) = self.current_image.take() {
                self.redo_stack.push(current);
            }
            self.current_image = Some(prev);
            true
        } else {
            false
        }
    }

    /// Redo previously undone action.
    pub fn redo(&mut self) -> bool {
        if let Some(next) = self.redo_stack.pop() {
            if let Some(current) = self.current_image.take() {
                self.undo_stack.push(current);
            }
            self.current_image = Some(next);
            true
        } else {
            false
        }
    }

    /// Export the current full-resolution image using the tiled pipeline.
    pub fn export_full_image(
        &self,
        format: lumora_core::exporter::ExportFormat,
        tile_size: usize,
    ) -> Result<Vec<u8>, String> {
        let img = self
            .current_image
            .as_ref()
            .ok_or_else(|| "No image currently loaded in engine".to_string())?;
        img.export_full_resolution(format, tile_size)
            .map_err(|e| e.to_string())
    }

    /// Dispatch and execute a Contract event by name.
    pub fn handle_event(&mut self, event_name: &str, payload_json: &str) -> EventResponse {
        let img = match self.current_image.as_mut() {
            Some(i) => i,
            None => {
                return EventResponse {
                    success: false,
                    event: event_name.to_string(),
                    width: 0,
                    height: 0,
                    error: Some("No image currently loaded in engine".to_string()),
                }
            }
        };

        match event_name {
            "update_exposure" => {
                #[derive(Deserialize)]
                struct Payload {
                    #[serde(alias = "val")]
                    ev: f32,
                }
                if let Ok(p) = serde_json::from_str::<Payload>(payload_json) {
                    self.push_undo();
                    if let Some(img) = self.current_image.as_mut() {
                        img.exposure = p.ev.clamp(-5.0, 5.0);
                        let _ = img.render_preview(1080, 1080);
                    }
                }
            }
            "update_contrast" => {
                #[derive(Deserialize)]
                struct Payload {
                    #[serde(alias = "val", alias = "contrast")]
                    factor: f32,
                }
                if let Ok(p) = serde_json::from_str::<Payload>(payload_json) {
                    self.push_undo();
                    if let Some(img) = self.current_image.as_mut() {
                        img.contrast = p.factor.clamp(-1.0, 1.0);
                        let _ = img.render_preview(1080, 1080);
                    }
                }
            }
            "update_saturation" => {
                #[derive(Deserialize)]
                struct Payload {
                    #[serde(alias = "val", alias = "saturation")]
                    factor: f32,
                }
                if let Ok(p) = serde_json::from_str::<Payload>(payload_json) {
                    self.push_undo();
                    if let Some(img) = self.current_image.as_mut() {
                        img.saturation = p.factor.clamp(-1.0, 1.0);
                        let _ = img.render_preview(1080, 1080);
                    }
                }
            }
            "curve_point_updated" => {
                #[derive(Deserialize)]
                struct Payload {
                    channel: String,
                    #[serde(default)]
                    points: Option<Vec<[f32; 2]>>,
                    #[serde(default)]
                    x: Option<f32>,
                    #[serde(default)]
                    y: Option<f32>,
                }
                if let Ok(p) = serde_json::from_str::<Payload>(payload_json) {
                    self.push_undo();
                    if let Some(img) = self.current_image.as_mut() {
                        let target_curve = match p.channel.as_str() {
                            "red" => &mut img.curves.red,
                            "green" => &mut img.curves.green,
                            "blue" => &mut img.curves.blue,
                            _ => &mut img.curves.master,
                        };
                        if let Some(pts) = p.points {
                            *target_curve =
                                pts.iter().map(|pt| CurvePoint::new(pt[0], pt[1])).collect();
                        } else if let (Some(x), Some(y)) = (p.x, p.y) {
                            target_curve.push(CurvePoint::new(x, y));
                            target_curve.sort_by(|a, b| {
                                a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal)
                            });
                        }
                        let _ = img.render_preview(1080, 1080);
                    }
                }
            }
            "update_hsl" => {
                #[derive(Deserialize)]
                struct Payload {
                    channel: usize,
                    hue: f32,
                    sat: f32,
                    lum: f32,
                }
                if let Ok(p) = serde_json::from_str::<Payload>(payload_json) {
                    self.push_undo();
                    if let Some(img) = self.current_image.as_mut() {
                        let adj = ChannelAdjust {
                            hue: p.hue,
                            saturation: p.sat,
                            luminance: p.lum,
                        };
                        match p.channel {
                            0 => img.hsl.red = adj,
                            1 => img.hsl.orange = adj,
                            2 => img.hsl.yellow = adj,
                            3 => img.hsl.green = adj,
                            4 => img.hsl.aqua = adj,
                            5 => img.hsl.blue = adj,
                            6 => img.hsl.purple = adj,
                            7 => img.hsl.magenta = adj,
                            _ => {}
                        }
                        let _ = img.render_preview(1080, 1080);
                    }
                }
            }
            "update_hsl_hue" => {
                #[derive(Deserialize)]
                struct Payload {
                    channel: usize,
                    val: f32,
                }
                if let Ok(p) = serde_json::from_str::<Payload>(payload_json) {
                    self.push_undo();
                    if let Some(img) = self.current_image.as_mut() {
                        let ch = match p.channel {
                            0 => &mut img.hsl.red,
                            1 => &mut img.hsl.orange,
                            2 => &mut img.hsl.yellow,
                            3 => &mut img.hsl.green,
                            4 => &mut img.hsl.aqua,
                            5 => &mut img.hsl.blue,
                            6 => &mut img.hsl.purple,
                            7 => &mut img.hsl.magenta,
                            _ => &mut img.hsl.red,
                        };
                        ch.hue = p.val.clamp(-30.0, 30.0);
                        let _ = img.render_preview(1080, 1080);
                    }
                }
            }
            "update_hsl_sat" => {
                #[derive(Deserialize)]
                struct Payload {
                    channel: usize,
                    val: f32,
                }
                if let Ok(p) = serde_json::from_str::<Payload>(payload_json) {
                    self.push_undo();
                    if let Some(img) = self.current_image.as_mut() {
                        let ch = match p.channel {
                            0 => &mut img.hsl.red,
                            1 => &mut img.hsl.orange,
                            2 => &mut img.hsl.yellow,
                            3 => &mut img.hsl.green,
                            4 => &mut img.hsl.aqua,
                            5 => &mut img.hsl.blue,
                            6 => &mut img.hsl.purple,
                            7 => &mut img.hsl.magenta,
                            _ => &mut img.hsl.red,
                        };
                        ch.saturation = p.val.clamp(-1.0, 1.0);
                        let _ = img.render_preview(1080, 1080);
                    }
                }
            }
            "update_hsl_lum" => {
                #[derive(Deserialize)]
                struct Payload {
                    channel: usize,
                    val: f32,
                }
                if let Ok(p) = serde_json::from_str::<Payload>(payload_json) {
                    self.push_undo();
                    if let Some(img) = self.current_image.as_mut() {
                        let ch = match p.channel {
                            0 => &mut img.hsl.red,
                            1 => &mut img.hsl.orange,
                            2 => &mut img.hsl.yellow,
                            3 => &mut img.hsl.green,
                            4 => &mut img.hsl.aqua,
                            5 => &mut img.hsl.blue,
                            6 => &mut img.hsl.purple,
                            7 => &mut img.hsl.magenta,
                            _ => &mut img.hsl.red,
                        };
                        ch.luminance = p.val.clamp(-1.0, 1.0);
                        let _ = img.render_preview(1080, 1080);
                    }
                }
            }
            "update_shadows_hue" => {
                #[derive(Deserialize)]
                struct Payload {
                    val: f32,
                }
                if let Ok(p) = serde_json::from_str::<Payload>(payload_json) {
                    self.push_undo();
                    if let Some(img) = self.current_image.as_mut() {
                        img.color_grading.shadows.hue = p.val.clamp(0.0, 360.0);
                        if img.color_grading.shadows.saturation < 0.05 {
                            img.color_grading.shadows.saturation = 0.25;
                        }
                        let _ = img.render_preview(1080, 1080);
                    }
                }
            }
            "update_highlights_hue" => {
                #[derive(Deserialize)]
                struct Payload {
                    val: f32,
                }
                if let Ok(p) = serde_json::from_str::<Payload>(payload_json) {
                    self.push_undo();
                    if let Some(img) = self.current_image.as_mut() {
                        img.color_grading.highlights.hue = p.val.clamp(0.0, 360.0);
                        if img.color_grading.highlights.saturation < 0.05 {
                            img.color_grading.highlights.saturation = 0.25;
                        }
                        let _ = img.render_preview(1080, 1080);
                    }
                }
            }
            "update_color_grading" => {
                #[derive(Deserialize)]
                struct Payload {
                    #[serde(default)]
                    shadows_h: f32,
                    #[serde(default)]
                    shadows_s: f32,
                    #[serde(default)]
                    highlights_h: f32,
                    #[serde(default)]
                    highlights_s: f32,
                }
                if let Ok(p) = serde_json::from_str::<Payload>(payload_json) {
                    self.push_undo();
                    if let Some(img) = self.current_image.as_mut() {
                        img.color_grading.shadows = GradingWheel {
                            hue: p.shadows_h.clamp(0.0, 360.0),
                            saturation: p.shadows_s.clamp(0.0, 1.0),
                            luminance: 0.0,
                        };
                        img.color_grading.highlights = GradingWheel {
                            hue: p.highlights_h.clamp(0.0, 360.0),
                            saturation: p.highlights_s.clamp(0.0, 1.0),
                            luminance: 0.0,
                        };
                        let _ = img.render_preview(1080, 1080);
                    }
                }
            }
            "add_new_upoint_pin" => {
                #[derive(Deserialize, Default)]
                struct Payload {
                    #[serde(default)]
                    x: Option<f32>,
                    #[serde(default)]
                    y: Option<f32>,
                    #[serde(default)]
                    radius: Option<f32>,
                    #[serde(default)]
                    brightness: Option<f32>,
                    #[serde(default)]
                    contrast: Option<f32>,
                    #[serde(default)]
                    saturation: Option<f32>,
                }
                let p = serde_json::from_str::<Payload>(payload_json).unwrap_or_default();
                self.push_undo();
                if let Some(img) = self.current_image.as_mut() {
                    let mut pin = UPointPin::new(
                        p.x.unwrap_or(0.5).clamp(0.0, 1.0),
                        p.y.unwrap_or(0.5).clamp(0.0, 1.0),
                        p.radius.unwrap_or(0.25).clamp(0.01, 2.0),
                    );
                    pin.brightness = p.brightness.unwrap_or(0.0).clamp(-1.0, 1.0);
                    pin.contrast = p.contrast.unwrap_or(0.0).clamp(-1.0, 1.0);
                    pin.saturation = p.saturation.unwrap_or(0.0).clamp(-1.0, 1.0);
                    img.upoints.add_pin(pin);
                    let _ = img.render_preview(1080, 1080);
                }
            }
            "update_pin_radius" => {
                #[derive(Deserialize)]
                struct Payload {
                    index: usize,
                    radius: f32,
                }
                if let Ok(p) = serde_json::from_str::<Payload>(payload_json) {
                    self.push_undo();
                    if let Some(img) = self.current_image.as_mut() {
                        if let Some(pin) = img.upoints.pins.get_mut(p.index) {
                            pin.radius = p.radius.clamp(0.01, 2.0);
                            let _ = img.render_preview(1080, 1080);
                        }
                    }
                }
            }
            "update_pin_brightness" => {
                #[derive(Deserialize)]
                struct Payload {
                    index: usize,
                    val: f32,
                }
                if let Ok(p) = serde_json::from_str::<Payload>(payload_json) {
                    self.push_undo();
                    if let Some(img) = self.current_image.as_mut() {
                        if let Some(pin) = img.upoints.pins.get_mut(p.index) {
                            pin.brightness = p.val.clamp(-1.0, 1.0);
                            let _ = img.render_preview(1080, 1080);
                        }
                    }
                }
            }
            "update_pin_contrast" => {
                #[derive(Deserialize)]
                struct Payload {
                    index: usize,
                    val: f32,
                }
                if let Ok(p) = serde_json::from_str::<Payload>(payload_json) {
                    self.push_undo();
                    if let Some(img) = self.current_image.as_mut() {
                        if let Some(pin) = img.upoints.pins.get_mut(p.index) {
                            pin.contrast = p.val.clamp(-1.0, 1.0);
                            let _ = img.render_preview(1080, 1080);
                        }
                    }
                }
            }
            "update_pin_saturation" => {
                #[derive(Deserialize)]
                struct Payload {
                    index: usize,
                    val: f32,
                }
                if let Ok(p) = serde_json::from_str::<Payload>(payload_json) {
                    self.push_undo();
                    if let Some(img) = self.current_image.as_mut() {
                        if let Some(pin) = img.upoints.pins.get_mut(p.index) {
                            pin.saturation = p.val.clamp(-1.0, 1.0);
                            let _ = img.render_preview(1080, 1080);
                        }
                    }
                }
            }
            "clear_all_upoint_pins" => {
                self.push_undo();
                if let Some(img) = self.current_image.as_mut() {
                    img.upoints.pins.clear();
                    let _ = img.render_preview(1080, 1080);
                }
            }
            "apply_ai_pass" => {
                #[derive(Deserialize)]
                struct Payload {
                    #[serde(alias = "type")]
                    op_type: i32,
                    strength: f32,
                }
                if let Ok(p) = serde_json::from_str::<Payload>(payload_json) {
                    self.push_undo();
                    if let Some(img) = self.current_image.as_mut() {
                        let clamped = p.strength.clamp(0.0, 1.0);
                        if let Some(existing) =
                            img.ai_passes.iter_mut().find(|(t, _)| *t == p.op_type)
                        {
                            existing.1 = clamped;
                        } else {
                            img.ai_passes.push((p.op_type, clamped));
                        }
                        let _ = img.render_preview(1080, 1080);
                    }
                }
            }
            "apply_scene_auto_enhance" => {
                #[derive(Deserialize)]
                struct Payload {
                    scene: String,
                }
                if let Ok(p) = serde_json::from_str::<Payload>(payload_json) {
                    self.push_undo();
                    if let Some(img) = self.current_image.as_mut() {
                        match p.scene.to_lowercase().as_str() {
                            "portrait" => {
                                img.exposure = 0.15;
                                img.contrast = 0.05;
                                img.hsl.red.saturation = 0.08;
                                img.hsl.orange.luminance = 0.10;
                            }
                            "landscape" => {
                                img.contrast = 0.18;
                                img.saturation = 0.12;
                                img.hsl.green.saturation = 0.20;
                                img.hsl.blue.saturation = 0.15;
                            }
                            "night" | "low_light" => {
                                img.exposure = 0.25;
                                if let Some(existing) =
                                    img.ai_passes.iter_mut().find(|(t, _)| *t == 2)
                                {
                                    existing.1 = 0.6;
                                } else {
                                    img.ai_passes.push((2, 0.6));
                                }
                            }
                            _ => {
                                img.exposure = 0.1;
                                img.contrast = 0.1;
                            }
                        }
                        let _ = img.render_preview(1080, 1080);
                    }
                }
            }
            "apply_preset_to_engine" => {
                #[derive(Deserialize)]
                struct Payload {
                    preset_id: u32,
                }
                if let Ok(p) = serde_json::from_str::<Payload>(payload_json) {
                    self.push_undo();
                    if let Some(img) = self.current_image.as_mut() {
                        match p.preset_id {
                            1 => {
                                // Portra 400 Warmth
                                img.exposure = 0.15;
                                img.contrast = 0.08;
                                img.saturation = 0.05;
                                img.color_grading.shadows.hue = 35.0;
                                img.color_grading.shadows.saturation = 0.15;
                            }
                            2 => {
                                // Nordic Moody Mist
                                img.exposure = -0.08;
                                img.contrast = 0.22;
                                img.saturation = -0.15;
                                img.color_grading.shadows.hue = 215.0;
                                img.color_grading.shadows.saturation = 0.25;
                            }
                            _ => {}
                        }
                        let _ = img.render_preview(1080, 1080);
                    }
                }
            }
            "apply_preset_json" => {
                #[derive(Deserialize)]
                struct WrappedPayload {
                    json: String,
                }
                let preset_opt = LumoraPreset::from_json(payload_json).ok().or_else(|| {
                    serde_json::from_str::<WrappedPayload>(payload_json)
                        .ok()
                        .and_then(|w| LumoraPreset::from_json(&w.json).ok())
                });

                if let Some(preset) = preset_opt {
                    self.push_undo();
                    if let Some(img) = self.current_image.as_mut() {
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
                        let _ = img.render_preview(1080, 1080);
                    }
                }
            }
            "export_rendered_image" => {
                // Renders full-resolution buffer without preview bounds
                if let Some(img) = self.current_image.as_mut() {
                    let _ = img.render_preview(0, 0);
                }
            }
            "undo_edit" => {
                self.undo();
                if let Some(img) = self.current_image.as_mut() {
                    let _ = img.render_preview(1080, 1080);
                }
            }
            "redo_edit" => {
                self.redo();
                if let Some(img) = self.current_image.as_mut() {
                    let _ = img.render_preview(1080, 1080);
                }
            }
            _ => {
                return EventResponse {
                    success: false,
                    event: event_name.to_string(),
                    width: img.preview.width as u32,
                    height: img.preview.height as u32,
                    error: Some(format!("Unknown event '{event_name}'")),
                }
            }
        }

        let cur = self.current_image.as_ref().unwrap();
        EventResponse {
            success: true,
            event: event_name.to_string(),
            width: cur.preview.width as u32,
            height: cur.preview.height as u32,
            error: None,
        }
    }

    /// Retrieve preview buffer data slice.
    pub fn get_preview_data(&self) -> Option<&[u8]> {
        self.current_image
            .as_ref()
            .map(|i| i.preview.data.as_slice())
    }

    /// Retrieve preview dimensions (width, height).
    pub fn get_preview_dimensions(&self) -> Option<(u32, u32)> {
        self.current_image
            .as_ref()
            .map(|i| (i.preview.width as u32, i.preview.height as u32))
    }
}
