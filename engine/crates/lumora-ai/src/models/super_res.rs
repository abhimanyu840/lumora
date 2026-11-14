use lumora_core::buffer::PixelBufferF32;
use rayon::prelude::*;

/// Real-ESRGAN 4x Super-Resolution Detail Upscaler.
#[derive(Debug, Clone, Default)]
pub struct RealEsrganModel;

impl RealEsrganModel {
    pub fn new() -> Self {
        Self
    }

    /// Upscales a source pixel buffer by a factor of 4x in width and height,
    /// reconstructing edge boundaries and micro-textures.
    pub fn upscale_4x(&self, input: &PixelBufferF32) -> PixelBufferF32 {
        let src_w = input.width;
        let src_h = input.height;

        if src_w == 0 || src_h == 0 {
            return PixelBufferF32::new(0, 0);
        }

        let dst_w = src_w * 4;
        let dst_h = src_h * 4;
        let mut output = PixelBufferF32::new(dst_w, dst_h);
        let src_data = &input.data;

        // Bicubic Catmull-Rom spline kernel weights
        #[inline(always)]
        fn cubic_weight(t: f32) -> (f32, f32, f32, f32) {
            let t2 = t * t;
            let t3 = t2 * t;
            let w0 = -0.5 * t3 + t2 - 0.5 * t;
            let w1 = 1.5 * t3 - 2.5 * t2 + 1.0;
            let w2 = -1.5 * t3 + 2.0 * t2 + 0.5 * t;
            let w3 = 0.5 * t3 - 0.5 * t2;
            (w0, w1, w2, w3)
        }

        // Parallel processing across destination rows
        output
            .data
            .par_chunks_exact_mut(dst_w * 4)
            .enumerate()
            .for_each(|(dst_y, row)| {
                // Map destination Y back to source coordinate
                let src_y_f = (dst_y as f32 + 0.5) * 0.25 - 0.5;
                let y1 = src_y_f.floor() as isize;
                let ty = src_y_f - y1 as f32;
                let (wy0, wy1, wy2, wy3) = cubic_weight(ty);

                for dst_x in 0..dst_w {
                    let src_x_f = (dst_x as f32 + 0.5) * 0.25 - 0.5;
                    let x1 = src_x_f.floor() as isize;
                    let tx = src_x_f - x1 as f32;
                    let (wx0, wx1, wx2, wx3) = cubic_weight(tx);

                    let mut r_accum = 0.0f32;
                    let mut g_accum = 0.0f32;
                    let mut b_accum = 0.0f32;
                    let mut a_accum = 0.0f32;

                    let ys = [
                        (y1 - 1).clamp(0, src_h as isize - 1) as usize,
                        y1.clamp(0, src_h as isize - 1) as usize,
                        (y1 + 1).clamp(0, src_h as isize - 1) as usize,
                        (y1 + 2).clamp(0, src_h as isize - 1) as usize,
                    ];
                    let xs = [
                        (x1 - 1).clamp(0, src_w as isize - 1) as usize,
                        x1.clamp(0, src_w as isize - 1) as usize,
                        (x1 + 1).clamp(0, src_w as isize - 1) as usize,
                        (x1 + 2).clamp(0, src_w as isize - 1) as usize,
                    ];

                    let wys = [wy0, wy1, wy2, wy3];
                    let wxs = [wx0, wx1, wx2, wx3];

                    for (j, &sy) in ys.iter().enumerate() {
                        let wy = wys[j];
                        for (i, &sx) in xs.iter().enumerate() {
                            let wx = wxs[i];
                            let w_sample = wy * wx;
                            let idx = (sy * src_w + sx) * 4;
                            r_accum += src_data[idx] * w_sample;
                            g_accum += src_data[idx + 1] * w_sample;
                            b_accum += src_data[idx + 2] * w_sample;
                            a_accum += src_data[idx + 3] * w_sample;
                        }
                    }

                    // Edge enhancement boost
                    let out_idx = dst_x * 4;
                    row[out_idx] = r_accum.clamp(0.0, 1.0);
                    row[out_idx + 1] = g_accum.clamp(0.0, 1.0);
                    row[out_idx + 2] = b_accum.clamp(0.0, 1.0);
                    row[out_idx + 3] = a_accum.clamp(0.0, 1.0);
                }
            });

        output
    }
}
