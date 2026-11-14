use lumora_core::buffer::PixelBufferF32;
use lumora_core::color::{srgb_to_lab, Lab};
use lumora_core::dag::ImageOp;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UPointPin {
    pub center_x: f32,           // Normalized coordinate 0.0 to 1.0
    pub center_y: f32,           // Normalized coordinate 0.0 to 1.0
    pub radius: f32,             // Normalized radius 0.01 to 2.0
    pub target_lab: Option<Lab>, // Target color sampled at center (or computed automatically)
    pub brightness: f32,         // -1.0 to 1.0
    pub contrast: f32,           // -1.0 to 1.0
    pub saturation: f32,         // -1.0 to 1.0
}

impl UPointPin {
    pub fn new(x: f32, y: f32, radius: f32) -> Self {
        Self {
            center_x: x.clamp(0.0, 1.0),
            center_y: y.clamp(0.0, 1.0),
            radius: radius.clamp(0.001, 2.0),
            target_lab: None,
            brightness: 0.0,
            contrast: 0.0,
            saturation: 0.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct UPointMask {
    pub pins: Vec<UPointPin>,
}

impl UPointMask {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_pin(&mut self, pin: UPointPin) {
        self.pins.push(pin);
    }

    /// Pre-resolve sampled target Lab values for all control pins from a reference buffer.
    pub fn resolve_targets(&mut self, buffer: &PixelBufferF32) {
        let width = buffer.width;
        let height = buffer.height;
        if width == 0 || height == 0 {
            return;
        }
        for pin in &mut self.pins {
            if pin.target_lab.is_none() {
                let px =
                    ((pin.center_x * (width - 1) as f32) as usize).min(width.saturating_sub(1));
                let py =
                    ((pin.center_y * (height - 1) as f32) as usize).min(height.saturating_sub(1));
                if let Some(p) = buffer.get_pixel(px, py) {
                    pin.target_lab = Some(srgb_to_lab(p.r, p.g, p.b));
                }
            }
        }
    }

    pub fn apply_to_buffer(&self, buffer: &mut PixelBufferF32) {
        self.apply_to_tile(buffer, 0, 0, buffer.width, buffer.height);
    }

    /// Apply selective U-Point pins to an extracted tile using global image coordinates.
    pub fn apply_to_tile(
        &self,
        tile: &mut PixelBufferF32,
        origin_x: usize,
        origin_y: usize,
        full_width: usize,
        full_height: usize,
    ) {
        if self.pins.is_empty()
            || tile.width == 0
            || tile.height == 0
            || full_width == 0
            || full_height == 0
        {
            return;
        }

        let tile_w = tile.width;
        let aspect = full_width as f32 / full_height as f32;

        // Ensure target Lab is sampled for pins that do not have it explicitly set
        let mut resolved_pins = self.pins.clone();
        for pin in &mut resolved_pins {
            if pin.target_lab.is_none() {
                let px = ((pin.center_x * (full_width - 1) as f32) as usize)
                    .min(full_width.saturating_sub(1));
                let py = ((pin.center_y * (full_height - 1) as f32) as usize)
                    .min(full_height.saturating_sub(1));
                if px >= origin_x
                    && px < origin_x + tile_w
                    && py >= origin_y
                    && py < origin_y + tile.height
                {
                    if let Some(p) = tile.get_pixel(px - origin_x, py - origin_y) {
                        pin.target_lab = Some(srgb_to_lab(p.r, p.g, p.b));
                    }
                }
            }
        }

        let sigma_sq = 2.0 * 0.5 * 0.5; // 2 * sigma^2 with sigma = 0.5
        let color_tolerance = 45.0f32; // CIE-Lab tolerance threshold

        // Process in parallel row-by-row using Rayon
        tile.data
            .par_chunks_exact_mut(tile_w * 4)
            .enumerate()
            .for_each(|(y, row)| {
                let global_y = origin_y + y;
                let norm_y = global_y as f32 / full_height as f32;

                for x in 0..tile_w {
                    let global_x = origin_x + x;
                    let norm_x = global_x as f32 / full_width as f32;
                    let offset = x * 4;

                    let mut cur_r = row[offset];
                    let mut cur_g = row[offset + 1];
                    let mut cur_b = row[offset + 2];

                    let mut pixel_lab: Option<Lab> = None;
                    let mut modified = false;

                    for pin in &resolved_pins {
                        let target_lab = match pin.target_lab {
                            Some(t) => t,
                            None => continue,
                        };

                        // 1. Fast Spatial Distance Check (aspect ratio compensated)
                        let dx = (norm_x - pin.center_x) * aspect;
                        let dy = norm_y - pin.center_y;
                        let d_spatial = (dx * dx + dy * dy).sqrt() / pin.radius;

                        if d_spatial > 1.0 {
                            continue;
                        }

                        // 2. Compute Lab color lazily only when pixel is within spatial influence of a pin
                        let lab = pixel_lab.get_or_insert_with(|| srgb_to_lab(cur_r, cur_g, cur_b));

                        // 3. Lab color similarity distance
                        let d_color = lab.delta_e_cie76(target_lab) / color_tolerance;

                        // 4. Combined Euclidean distance
                        let d_combined = (d_spatial * d_spatial + d_color * d_color).sqrt();
                        if d_combined > 1.8 {
                            continue;
                        }

                        // 5. Nik exponential falloff weight
                        let weight = (-d_combined * d_combined / sigma_sq).exp();
                        if weight < 0.005 {
                            continue;
                        }

                        modified = true;

                        // Accumulate selective brightness
                        cur_r += pin.brightness * weight * 0.5;
                        cur_g += pin.brightness * weight * 0.5;
                        cur_b += pin.brightness * weight * 0.5;

                        // Accumulate selective contrast
                        if pin.contrast.abs() > 1e-4 {
                            let c_factor = 1.0 + pin.contrast * weight;
                            cur_r = (cur_r - 0.5) * c_factor + 0.5;
                            cur_g = (cur_g - 0.5) * c_factor + 0.5;
                            cur_b = (cur_b - 0.5) * c_factor + 0.5;
                        }

                        // Accumulate selective saturation
                        if pin.saturation.abs() > 1e-4 {
                            let lum = 0.2126 * cur_r + 0.7152 * cur_g + 0.0722 * cur_b;
                            let s_mult = 1.0 + pin.saturation * weight;
                            cur_r = lum + (cur_r - lum) * s_mult;
                            cur_g = lum + (cur_g - lum) * s_mult;
                            cur_b = lum + (cur_b - lum) * s_mult;
                        }
                    }

                    if modified {
                        row[offset] = cur_r.clamp(0.0, 1.0);
                        row[offset + 1] = cur_g.clamp(0.0, 1.0);
                        row[offset + 2] = cur_b.clamp(0.0, 1.0);
                    }
                }
            });
    }
}

impl ImageOp for UPointMask {
    fn name(&self) -> &str {
        "UPointMask"
    }

    fn apply(&self, buffer: &mut PixelBufferF32) {
        self.apply_to_buffer(buffer);
    }

    fn apply_tile(
        &self,
        tile: &mut PixelBufferF32,
        origin_x: usize,
        origin_y: usize,
        full_width: usize,
        full_height: usize,
    ) {
        self.apply_to_tile(tile, origin_x, origin_y, full_width, full_height);
    }
}
