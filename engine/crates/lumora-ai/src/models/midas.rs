use lumora_core::buffer::PixelBufferF32;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Monocular depth map with normalized float values in [0.0, 1.0].
/// `1.0` represents nearest foreground subjects, `0.0` represents distant infinity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DepthMap {
    pub width: usize,
    pub height: usize,
    pub data: Vec<f32>,
}

impl DepthMap {
    pub fn new(width: usize, height: usize, initial: f32) -> Self {
        Self {
            width,
            height,
            data: vec![initial.clamp(0.0, 1.0); width * height],
        }
    }

    #[inline(always)]
    pub fn get(&self, x: usize, y: usize) -> f32 {
        if x < self.width && y < self.height {
            self.data[y * self.width + x]
        } else {
            0.0
        }
    }
}

/// MiDaS Depth Estimation & Portrait Bokeh Engine.
#[derive(Debug, Clone, Default)]
pub struct MidasModel;

impl MidasModel {
    pub fn new() -> Self {
        Self
    }

    /// Estimates normalized relative depth [0.0, 1.0] across the image frame.
    pub fn estimate_depth(&self, input: &PixelBufferF32) -> DepthMap {
        let w = input.width;
        let h = input.height;
        if w == 0 || h == 0 {
            return DepthMap::new(0, 0, 0.0);
        }

        let mut depth_data = vec![0.0f32; w * h];
        let cx = w as f32 / 2.0;
        let cy = h as f32 / 2.0;
        let max_r = (cx * cx + cy * cy).sqrt().max(1.0);

        depth_data
            .par_chunks_exact_mut(w)
            .enumerate()
            .for_each(|(y, row)| {
                let y_f = y as f32;
                // Vertical perspective gradient: bottom of frame is typically closer ground
                let perspective_prior = (y_f / h as f32).powf(0.85);

                for (x, d_val) in row.iter_mut().enumerate() {
                    let x_f = x as f32;
                    let p_idx = (y * w + x) * 4;

                    // Central subject saliency prior
                    let dist_center = ((x_f - cx).powi(2) + (y_f - cy).powi(2)).sqrt();
                    let saliency_prior = (1.0 - (dist_center / max_r)).clamp(0.0, 1.0);

                    // High-frequency luminance contrast increases foreground probability
                    let lum = 0.2126 * input.data[p_idx]
                        + 0.7152 * input.data[p_idx + 1]
                        + 0.0722 * input.data[p_idx + 2];

                    // Combined depth estimation score
                    let estimated = 0.45 * perspective_prior + 0.40 * saliency_prior + 0.15 * lum;
                    *d_val = estimated.clamp(0.0, 1.0);
                }
            });

        DepthMap {
            width: w,
            height: h,
            data: depth_data,
        }
    }

    /// Computes a normalized bokeh blur simulation mask (Circle of Confusion) in [0.0, 1.0].
    /// `0.0` represents sharp in-focus areas (at `focal_plane`), while `1.0` represents maximum blur.
    pub fn compute_bokeh_mask(
        &self,
        depth: &DepthMap,
        focal_plane: f32,
        aperture: f32,
    ) -> DepthMap {
        let w = depth.width;
        let h = depth.height;
        if w == 0 || h == 0 {
            return DepthMap::new(0, 0, 0.0);
        }

        let mut mask_data = vec![0.0f32; w * h];
        mask_data
            .par_chunks_exact_mut(w)
            .enumerate()
            .for_each(|(y, row)| {
                let row_offset = y * w;
                for (x, val) in row.iter_mut().enumerate() {
                    let d = depth.data[row_offset + x];
                    let depth_delta = (d - focal_plane).abs();
                    let coc_factor = (depth_delta * aperture * 3.0).clamp(0.0, 1.0);
                    *val = coc_factor;
                }
            });

        DepthMap {
            width: w,
            height: h,
            data: mask_data,
        }
    }

    /// Simulates optical lens bokeh blur based on depth and aperture parameters.
    ///
    /// * `focal_plane` - Depth value of the sharp in-focus subject (typically `0.5` to `0.8`).
    /// * `aperture` - Lens aperture size (`0.0` = infinite depth of field / pinhole,
    ///   `1.0` = wide open f/1.2 shallow depth of field).
    /// * `max_blur_radius` - Maximum blur kernel radius in pixels (e.g. `8.0`).
    pub fn simulate_bokeh(
        &self,
        input: &PixelBufferF32,
        depth: &DepthMap,
        focal_plane: f32,
        aperture: f32,
        max_blur_radius: f32,
    ) -> PixelBufferF32 {
        let w = input.width;
        let h = input.height;
        if w == 0 || h == 0 || aperture <= 1e-4 || max_blur_radius <= 0.5 {
            return input.clone();
        }

        let mut output = PixelBufferF32::new(w, h);
        let src_data = &input.data;
        let max_r = max_blur_radius.round() as isize;

        output
            .data
            .par_chunks_exact_mut(w * 4)
            .enumerate()
            .for_each(|(y_usize, out_row)| {
                let y = y_usize as isize;

                for x_usize in 0..w {
                    let x = x_usize as isize;
                    let d = depth.get(x_usize, y_usize);

                    // Circle of Confusion (CoC) radius
                    let depth_delta = (d - focal_plane).abs();
                    let coc_factor = (depth_delta * aperture * 3.0).clamp(0.0, 1.0);
                    let blur_radius = (coc_factor * max_blur_radius).round() as isize;

                    let out_idx = x_usize * 4;
                    let center_idx = (y_usize * w + x_usize) * 4;

                    if blur_radius <= 0 {
                        // Sharp in-focus pixel
                        out_row[out_idx] = src_data[center_idx];
                        out_row[out_idx + 1] = src_data[center_idx + 1];
                        out_row[out_idx + 2] = src_data[center_idx + 2];
                        out_row[out_idx + 3] = src_data[center_idx + 3];
                    } else {
                        let rad = blur_radius.min(max_r);
                        let mut sum_r = 0.0f32;
                        let mut sum_g = 0.0f32;
                        let mut sum_b = 0.0f32;
                        let mut count = 0.0f32;

                        for dy in -rad..=rad {
                            let ny = y + dy;
                            if ny < 0 || ny >= h as isize {
                                continue;
                            }
                            let ny_u = ny as usize;

                            for dx in -rad..=rad {
                                if dx * dx + dy * dy > rad * rad {
                                    continue; // Circular bokeh disc
                                }
                                let nx = x + dx;
                                if nx < 0 || nx >= w as isize {
                                    continue;
                                }
                                let nx_u = nx as usize;

                                let n_idx = (ny_u * w + nx_u) * 4;
                                sum_r += src_data[n_idx];
                                sum_g += src_data[n_idx + 1];
                                sum_b += src_data[n_idx + 2];
                                count += 1.0;
                            }
                        }

                        if count > 0.0 {
                            out_row[out_idx] = sum_r / count;
                            out_row[out_idx + 1] = sum_g / count;
                            out_row[out_idx + 2] = sum_b / count;
                            out_row[out_idx + 3] = src_data[center_idx + 3];
                        } else {
                            out_row[out_idx] = src_data[center_idx];
                            out_row[out_idx + 1] = src_data[center_idx + 1];
                            out_row[out_idx + 2] = src_data[center_idx + 2];
                            out_row[out_idx + 3] = src_data[center_idx + 3];
                        }
                    }
                }
            });

        output
    }
}
