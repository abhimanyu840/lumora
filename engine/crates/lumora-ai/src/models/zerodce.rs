use lumora_core::buffer::PixelBufferF32;
use rayon::prelude::*;

/// Zero-Reference Deep Curve Estimation (Zero-DCE) Low-Light Enhancement Model.
///
/// Implements the 8-iteration non-linear quadratic curve mapping:
/// `LE_n(x) = LE_{n-1}(x) + alpha * A_n(x) * LE_{n-1}(x) * (1.0 - LE_{n-1}(x))`
///
/// Illuminates underexposed shadows without clipping highlights or amplifying noise.
#[derive(Debug, Clone, Default)]
pub struct ZeroDceModel;

impl ZeroDceModel {
    pub fn new() -> Self {
        Self
    }

    /// Enhances a low-light image using the Zero-DCE non-linear curve pipeline.
    ///
    /// * `intensity` - Multiplier for curve parameters, typically `1.0` (`0.0` = no change).
    pub fn enhance(&self, input: &PixelBufferF32, intensity: f32) -> PixelBufferF32 {
        let alpha = intensity.clamp(0.0, 2.5);
        if alpha <= 1e-5 {
            return input.clone();
        }

        let w = input.width;
        let h = input.height;
        if w == 0 || h == 0 {
            return input.clone();
        }

        let mut output = PixelBufferF32::new(w, h);
        let src_data = &input.data;

        // Number of non-linear curve iterations (standard Zero-DCE paper specifies 8)
        const NUM_ITERATIONS: usize = 8;

        // Process in parallel across row chunks
        output
            .data
            .par_chunks_exact_mut(w * 4)
            .enumerate()
            .for_each(|(y, row)| {
                let row_offset = y * w * 4;
                for x in 0..w {
                    let idx = x * 4;
                    let src_idx = row_offset + idx;

                    let mut r = src_data[src_idx];
                    let mut g = src_data[src_idx + 1];
                    let mut b = src_data[src_idx + 2];
                    let a = src_data[src_idx + 3];

                    // Pixel luminance in [0.0, 1.0]
                    let lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;

                    // Estimation of curve parameter A based on local illumination deficiency.
                    // Darker regions need stronger upward curvature:
                    // A is positive in underexposed areas and tapers off in well-exposed highlights.
                    let a_param = ((1.0 - lum).powf(1.4) * 0.45 * alpha).clamp(-1.0, 1.0);

                    // Apply the 8 successive quadratic curve iterations
                    for _ in 0..NUM_ITERATIONS {
                        r = r + a_param * r * (1.0 - r);
                        g = g + a_param * g * (1.0 - g);
                        b = b + a_param * b * (1.0 - b);
                    }

                    row[idx] = r.clamp(0.0, 1.0);
                    row[idx + 1] = g.clamp(0.0, 1.0);
                    row[idx + 2] = b.clamp(0.0, 1.0);
                    row[idx + 3] = a;
                }
            });

        output
    }
}
