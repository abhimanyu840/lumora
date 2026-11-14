use lumora_core::buffer::PixelBufferF32;
use lumora_core::color::{hsl_to_rgb, Hsl};
use lumora_core::dag::ImageOp;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

const CHUNK_FLOATS: usize = 1024 * 4;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GradingWheel {
    pub hue: f32,        // 0.0 to 360.0
    pub saturation: f32, // 0.0 to 1.0
    pub luminance: f32,  // -1.0 to 1.0
}

impl Default for GradingWheel {
    fn default() -> Self {
        Self {
            hue: 0.0,
            saturation: 0.0,
            luminance: 0.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ColorGrading {
    pub shadows: GradingWheel,
    pub midtones: GradingWheel,
    pub highlights: GradingWheel,
    pub balance: f32,  // -1.0 to 1.0 (default 0.0)
    pub blending: f32, // 0.0 to 1.0 (default 0.5)
}

impl Default for ColorGrading {
    fn default() -> Self {
        Self {
            shadows: GradingWheel::default(),
            midtones: GradingWheel::default(),
            highlights: GradingWheel::default(),
            balance: 0.0,
            blending: 0.5,
        }
    }
}

impl ColorGrading {
    pub fn apply_to_buffer(&self, buffer: &mut PixelBufferF32) {
        let is_noop = self.shadows.saturation < 1e-4
            && self.shadows.luminance.abs() < 1e-4
            && self.midtones.saturation < 1e-4
            && self.midtones.luminance.abs() < 1e-4
            && self.highlights.saturation < 1e-4
            && self.highlights.luminance.abs() < 1e-4;

        if is_noop {
            return;
        }

        // Precompute tint colors for each wheel
        let tint_s = hsl_to_rgb(Hsl::new(self.shadows.hue, 1.0, 0.5));
        let tint_m = hsl_to_rgb(Hsl::new(self.midtones.hue, 1.0, 0.5));
        let tint_h = hsl_to_rgb(Hsl::new(self.highlights.hue, 1.0, 0.5));

        let balance_shift = self.balance.clamp(-1.0, 1.0) * 0.25;
        let blend_exp = 1.0 / (self.blending.clamp(0.1, 1.0) * 2.0);
        let is_linear_blend = (blend_exp - 1.0).abs() < 1e-4;

        buffer.data.par_chunks_mut(CHUNK_FLOATS).for_each(|slice| {
            for chunk in slice.chunks_exact_mut(4) {
                let r = chunk[0];
                let g = chunk[1];
                let b = chunk[2];

                // Rec.709 luminance
                let lum = (0.2126 * r + 0.7152 * g + 0.0722 * b - balance_shift).clamp(0.0, 1.0);

                // Compute split weights
                let (ws_raw, wh_raw, wm_raw) = if is_linear_blend {
                    (
                        (1.0 - lum / 0.5).max(0.0),
                        ((lum - 0.5) / 0.5).max(0.0),
                        (1.0 - ((lum - 0.5).abs() / 0.5)).max(0.0),
                    )
                } else {
                    (
                        (1.0 - lum / 0.5).max(0.0).powf(blend_exp),
                        ((lum - 0.5) / 0.5).max(0.0).powf(blend_exp),
                        (1.0 - ((lum - 0.5).abs() / 0.5)).max(0.0).powf(blend_exp),
                    )
                };

                let total_w = ws_raw + wm_raw + wh_raw;
                if total_w < 1e-5 {
                    continue;
                }

                let ws = ws_raw / total_w;
                let wm = wm_raw / total_w;
                let wh = wh_raw / total_w;

                // Apply color tint offsets
                let tint_r = ws * self.shadows.saturation * (tint_s.0 - 0.5) * 2.0
                    + wm * self.midtones.saturation * (tint_m.0 - 0.5) * 2.0
                    + wh * self.highlights.saturation * (tint_h.0 - 0.5) * 2.0;

                let tint_g = ws * self.shadows.saturation * (tint_s.1 - 0.5) * 2.0
                    + wm * self.midtones.saturation * (tint_m.1 - 0.5) * 2.0
                    + wh * self.highlights.saturation * (tint_h.1 - 0.5) * 2.0;

                let tint_b = ws * self.shadows.saturation * (tint_s.2 - 0.5) * 2.0
                    + wm * self.midtones.saturation * (tint_m.2 - 0.5) * 2.0
                    + wh * self.highlights.saturation * (tint_h.2 - 0.5) * 2.0;

                // Apply luminance offsets
                let lum_offset = ws * self.shadows.luminance
                    + wm * self.midtones.luminance
                    + wh * self.highlights.luminance;

                chunk[0] = (r + tint_r * 0.3 + lum_offset * 0.2).clamp(0.0, 1.0);
                chunk[1] = (g + tint_g * 0.3 + lum_offset * 0.2).clamp(0.0, 1.0);
                chunk[2] = (b + tint_b * 0.3 + lum_offset * 0.2).clamp(0.0, 1.0);
            }
        });
    }
}

impl ImageOp for ColorGrading {
    fn name(&self) -> &str {
        "ColorGrading"
    }

    fn apply(&self, buffer: &mut PixelBufferF32) {
        self.apply_to_buffer(buffer);
    }
}
