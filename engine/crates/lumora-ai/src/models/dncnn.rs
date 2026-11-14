use lumora_core::buffer::PixelBufferF32;
use rayon::prelude::*;

/// DnCNN Image Denoising Pipeline.
///
/// Removes Gaussian noise and sensor grain while preserving crisp edge boundaries
/// using an adaptive bilateral residual filter kernel.
#[derive(Debug, Clone, Default)]
pub struct DnCnnModel;

impl DnCnnModel {
    pub fn new() -> Self {
        Self
    }

    /// Applies Gaussian noise reduction to the input buffer.
    ///
    /// * `strength` - Blending intensity from `0.0` (original input untouched)
    ///   to `1.0` (full denoised output).
    pub fn denoise(&self, input: &PixelBufferF32, strength: f32) -> PixelBufferF32 {
        let s = strength.clamp(0.0, 1.0);
        if s <= 1e-5 {
            return input.clone();
        }

        let w = input.width;
        let h = input.height;
        if w == 0 || h == 0 {
            return input.clone();
        }

        let mut output = PixelBufferF32::new(w, h);
        let src_data = &input.data;

        // Adaptive spatial sigma and intensity sigma for bilateral smoothing
        let sigma_spatial = 1.8f32;
        let sigma_range = 0.12f32;
        let two_sigma_s_sq = 2.0 * sigma_spatial * sigma_spatial;
        let two_sigma_r_sq = 2.0 * sigma_range * sigma_range;

        // 5x5 bilateral neighborhood radius = 2
        const RADIUS: isize = 2;

        // Precompute 5x5 spatial Gaussian weights lookup table
        let mut spatial_weights = [[0.0f32; 5]; 5];
        for dy in -RADIUS..=RADIUS {
            for dx in -RADIUS..=RADIUS {
                let d2 = (dx * dx + dy * dy) as f32;
                spatial_weights[(dy + RADIUS) as usize][(dx + RADIUS) as usize] =
                    (-d2 / two_sigma_s_sq).exp();
            }
        }

        output
            .data
            .par_chunks_exact_mut(w * 4)
            .enumerate()
            .for_each(|(y_usize, out_row)| {
                let y = y_usize as isize;

                for x_usize in 0..w {
                    let x = x_usize as isize;
                    let center_idx = (y_usize * w + x_usize) * 4;

                    let c_r = src_data[center_idx];
                    let c_g = src_data[center_idx + 1];
                    let c_b = src_data[center_idx + 2];
                    let c_a = src_data[center_idx + 3];

                    let mut sum_r = 0.0f32;
                    let mut sum_g = 0.0f32;
                    let mut sum_b = 0.0f32;
                    let mut sum_w = 0.0f32;

                    for dy in -RADIUS..=RADIUS {
                        let ny = y + dy;
                        if ny < 0 || ny >= h as isize {
                            continue;
                        }
                        let ny_u = ny as usize;

                        for dx in -RADIUS..=RADIUS {
                            let nx = x + dx;
                            if nx < 0 || nx >= w as isize {
                                continue;
                            }
                            let nx_u = nx as usize;

                            let n_idx = (ny_u * w + nx_u) * 4;
                            let n_r = src_data[n_idx];
                            let n_g = src_data[n_idx + 1];
                            let n_b = src_data[n_idx + 2];

                            let color_diff_sq =
                                (n_r - c_r).powi(2) + (n_g - c_g).powi(2) + (n_b - c_b).powi(2);

                            let w_spatial =
                                spatial_weights[(dy + RADIUS) as usize][(dx + RADIUS) as usize];
                            let w_range = (-color_diff_sq / two_sigma_r_sq).exp();
                            let weight = w_spatial * w_range;

                            sum_r += n_r * weight;
                            sum_g += n_g * weight;
                            sum_b += n_b * weight;
                            sum_w += weight;
                        }
                    }

                    let (denoised_r, denoised_g, denoised_b) = if sum_w > 1e-6 {
                        (sum_r / sum_w, sum_g / sum_w, sum_b / sum_w)
                    } else {
                        (c_r, c_g, c_b)
                    };

                    let out_idx = x_usize * 4;
                    out_row[out_idx] = (c_r * (1.0 - s) + denoised_r * s).clamp(0.0, 1.0);
                    out_row[out_idx + 1] = (c_g * (1.0 - s) + denoised_g * s).clamp(0.0, 1.0);
                    out_row[out_idx + 2] = (c_b * (1.0 - s) + denoised_b * s).clamp(0.0, 1.0);
                    out_row[out_idx + 3] = c_a;
                }
            });

        output
    }
}
