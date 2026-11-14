use lumora_core::buffer::PixelBufferF32;
use lumora_core::color::{hsl_to_rgb, rgb_to_hsl, Hsl};
use lumora_core::dag::ImageOp;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

const CHUNK_FLOATS: usize = 1024 * 4;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ChannelAdjust {
    pub hue: f32,        // -30.0 to +30.0 degrees shift
    pub saturation: f32, // -1.0 to +1.0
    pub luminance: f32,  // -1.0 to +1.0
}

impl Default for ChannelAdjust {
    fn default() -> Self {
        Self {
            hue: 0.0,
            saturation: 0.0,
            luminance: 0.0,
        }
    }
}

pub const CHANNEL_CENTERS: [f32; 8] = [
    0.0,   // Red
    30.0,  // Orange
    60.0,  // Yellow
    120.0, // Green
    180.0, // Aqua
    240.0, // Blue
    285.0, // Purple
    325.0, // Magenta
];

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct HslMixer {
    pub red: ChannelAdjust,
    pub orange: ChannelAdjust,
    pub yellow: ChannelAdjust,
    pub green: ChannelAdjust,
    pub aqua: ChannelAdjust,
    pub blue: ChannelAdjust,
    pub purple: ChannelAdjust,
    pub magenta: ChannelAdjust,
}

impl HslMixer {
    pub fn channels(&self) -> [&ChannelAdjust; 8] {
        [
            &self.red,
            &self.orange,
            &self.yellow,
            &self.green,
            &self.aqua,
            &self.blue,
            &self.purple,
            &self.magenta,
        ]
    }

    #[inline(always)]
    fn circular_dist(h1: f32, h2: f32) -> f32 {
        let diff = (h1 - h2).abs() % 360.0;
        if diff > 180.0 {
            360.0 - diff
        } else {
            diff
        }
    }

    pub fn apply_to_buffer(&self, buffer: &mut PixelBufferF32) {
        let channels = self.channels();
        let is_noop = channels
            .iter()
            .all(|c| c.hue.abs() < 1e-4 && c.saturation.abs() < 1e-4 && c.luminance.abs() < 1e-4);
        if is_noop {
            return;
        }

        // Precompute 360-entry hue shift table for fast linear lookup
        let mut hue_table = [(0.0f32, 0.0f32, 0.0f32); 360];
        for (h_deg, entry) in hue_table.iter_mut().enumerate() {
            let h = h_deg as f32;
            let mut total_w = 0.0f32;
            let mut weights = [0.0f32; 8];

            for (i, &center) in CHANNEL_CENTERS.iter().enumerate() {
                let dist = Self::circular_dist(h, center);
                let bandwidth = 45.0; // Influenced within 45 degrees
                if dist < bandwidth {
                    let w = 1.0 - (dist / bandwidth);
                    let w_smooth = w * w * (3.0 - 2.0 * w); // Smoothstep
                    weights[i] = w_smooth;
                    total_w += w_smooth;
                }
            }

            if total_w >= 1e-5 {
                let mut d_hue = 0.0f32;
                let mut d_sat = 0.0f32;
                let mut d_lum = 0.0f32;

                for (i, &w) in weights.iter().enumerate() {
                    if w > 0.0 {
                        let norm_w = w / total_w;
                        d_hue += norm_w * channels[i].hue;
                        d_sat += norm_w * channels[i].saturation;
                        d_lum += norm_w * channels[i].luminance;
                    }
                }
                *entry = (d_hue, d_sat, d_lum);
            }
        }

        buffer.data.par_chunks_mut(CHUNK_FLOATS).for_each(|slice| {
            for chunk in slice.chunks_exact_mut(4) {
                let hsl = rgb_to_hsl(chunk[0], chunk[1], chunk[2]);
                if hsl.s < 0.01 {
                    // Achromatic pixels are unaffected by color hue/sat shifts
                    continue;
                }

                let (d_hue, d_sat, d_lum) = {
                    let h_clamped = hsl.h.clamp(0.0, 359.999);
                    let h_deg = h_clamped as usize;
                    let next_deg = (h_deg + 1) % 360;
                    let frac = h_clamped - h_deg as f32;
                    let e0 = hue_table[h_deg];
                    let e1 = hue_table[next_deg];
                    (
                        e0.0 * (1.0 - frac) + e1.0 * frac,
                        e0.1 * (1.0 - frac) + e1.1 * frac,
                        e0.2 * (1.0 - frac) + e1.2 * frac,
                    )
                };

                if d_hue.abs() < 1e-5 && d_sat.abs() < 1e-5 && d_lum.abs() < 1e-5 {
                    continue;
                }

                let new_h = (hsl.h + d_hue).rem_euclid(360.0);
                let new_s = if d_sat >= 0.0 {
                    hsl.s + (1.0 - hsl.s) * d_sat
                } else {
                    hsl.s * (1.0 + d_sat)
                }
                .clamp(0.0, 1.0);

                let new_l = if d_lum >= 0.0 {
                    hsl.l + (1.0 - hsl.l) * d_lum * 0.5
                } else {
                    hsl.l * (1.0 + d_lum * 0.5)
                }
                .clamp(0.0, 1.0);

                let (r, g, b) = hsl_to_rgb(Hsl::new(new_h, new_s, new_l));
                chunk[0] = r;
                chunk[1] = g;
                chunk[2] = b;
            }
        });
    }
}

impl ImageOp for HslMixer {
    fn name(&self) -> &str {
        "HslMixer"
    }

    fn apply(&self, buffer: &mut PixelBufferF32) {
        self.apply_to_buffer(buffer);
    }
}
