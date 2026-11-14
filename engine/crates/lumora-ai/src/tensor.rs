use lumora_core::buffer::PixelBufferF32;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Tensor {
    pub shape: Vec<usize>, // e.g. [batch, channels, height, width]
    pub data: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorError {
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    InvalidDimensions(String),
}

impl std::fmt::Display for TensorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ShapeMismatch { expected, actual } => {
                write!(
                    f,
                    "Tensor shape mismatch: expected {expected:?}, got {actual:?}"
                )
            }
            Self::InvalidDimensions(msg) => write!(f, "Invalid tensor dimensions: {msg}"),
        }
    }
}

impl std::error::Error for TensorError {}

impl Tensor {
    /// Creates a new tensor filled with zeros with the given shape.
    pub fn zeros(shape: Vec<usize>) -> Self {
        let size = shape.iter().product();
        Self {
            shape,
            data: vec![0.0f32; size],
        }
    }

    /// Creates a new tensor from raw float data and shape.
    pub fn from_data(shape: Vec<usize>, data: Vec<f32>) -> Result<Self, TensorError> {
        let expected: usize = shape.iter().product();
        if data.len() != expected {
            return Err(TensorError::ShapeMismatch {
                expected: vec![expected],
                actual: vec![data.len()],
            });
        }
        Ok(Self { shape, data })
    }

    /// Number of elements in the tensor.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the tensor contains no elements.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Number of dimensions (rank) of the tensor.
    #[inline(always)]
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Converts a Lumora `PixelBufferF32` (HWC format: height x width x 4 RGBA)
    /// to an NCHW tensor: [1, 3, height, width] (RGB channels only, omitting Alpha).
    pub fn from_pixel_buffer(buffer: &PixelBufferF32) -> Self {
        let w = buffer.width;
        let h = buffer.height;
        let plane_size = w * h;
        let mut data = vec![0.0f32; 3 * plane_size];

        let (r_plane, rest) = data.split_at_mut(plane_size);
        let (g_plane, b_plane) = rest.split_at_mut(plane_size);

        // Populate in parallel across rows
        r_plane
            .par_chunks_exact_mut(w)
            .zip(g_plane.par_chunks_exact_mut(w))
            .zip(b_plane.par_chunks_exact_mut(w))
            .enumerate()
            .for_each(|(y, ((r_row, g_row), b_row))| {
                let row_offset = y * w * 4;
                let src = &buffer.data[row_offset..row_offset + w * 4];
                for (x, chunk) in src.chunks_exact(4).enumerate() {
                    r_row[x] = chunk[0];
                    g_row[x] = chunk[1];
                    b_row[x] = chunk[2];
                }
            });

        Self {
            shape: vec![1, 3, h, w],
            data,
        }
    }

    /// Converts an NCHW tensor [1, 3, height, width], [3, height, width],
    /// or single-channel [1, 1, height, width] / [height, width]
    /// back into a Lumora `PixelBufferF32` (HWC format: height x width x 4 RGBA),
    /// clamping RGB values to [0.0, 1.0] and setting Alpha to 1.0.
    pub fn to_pixel_buffer(&self) -> Result<PixelBufferF32, TensorError> {
        let (channels, h, w) = match self.shape.as_slice() {
            [1, c, h, w] => (*c, *h, *w),
            [c, h, w] => (*c, *h, *w),
            [h, w] => (1, *h, *w),
            _ => {
                return Err(TensorError::InvalidDimensions(format!(
                    "Expected 2D, 3D or 4D tensor with 1 or 3 channels, got shape {:?}",
                    self.shape
                )))
            }
        };

        if channels != 3 && channels != 1 {
            return Err(TensorError::InvalidDimensions(format!(
                "Expected 1 or 3 channels, got {channels}"
            )));
        }

        let plane_size = w * h;
        let mut buffer = PixelBufferF32::new(w, h);

        if channels == 3 {
            let r_plane = &self.data[0..plane_size];
            let g_plane = &self.data[plane_size..2 * plane_size];
            let b_plane = &self.data[2 * plane_size..3 * plane_size];

            buffer
                .data
                .par_chunks_exact_mut(w * 4)
                .enumerate()
                .for_each(|(y, row)| {
                    let plane_offset = y * w;
                    for x in 0..w {
                        let idx = x * 4;
                        let p_idx = plane_offset + x;
                        row[idx] = r_plane[p_idx].clamp(0.0, 1.0);
                        row[idx + 1] = g_plane[p_idx].clamp(0.0, 1.0);
                        row[idx + 2] = b_plane[p_idx].clamp(0.0, 1.0);
                        row[idx + 3] = 1.0;
                    }
                });
        } else {
            // Single channel (e.g. depth or grayscale mask): duplicate across RGB
            let mono_plane = &self.data[0..plane_size];
            buffer
                .data
                .par_chunks_exact_mut(w * 4)
                .enumerate()
                .for_each(|(y, row)| {
                    let plane_offset = y * w;
                    for x in 0..w {
                        let idx = x * 4;
                        let v = mono_plane[plane_offset + x].clamp(0.0, 1.0);
                        row[idx] = v;
                        row[idx + 1] = v;
                        row[idx + 2] = v;
                        row[idx + 3] = 1.0;
                    }
                });
        }

        Ok(buffer)
    }

    /// Normalizes tensor channels using mean and standard deviation:
    /// `x_norm = (x - mean[c]) / std[c]`
    pub fn normalize(&mut self, mean: &[f32; 3], std: &[f32; 3]) -> Result<(), TensorError> {
        let (c, h, w) = self.spatial_dimensions_3ch()?;
        let plane_size = h * w;

        for ch in 0..c {
            let m = mean[ch];
            let s = std[ch];
            let start = ch * plane_size;
            let end = start + plane_size;
            let slice = &mut self.data[start..end];
            slice.par_iter_mut().for_each(|val| {
                *val = (*val - m) / s;
            });
        }
        Ok(())
    }

    /// Reverses mean/std normalization:
    /// `x = x_norm * std[c] + mean[c]`
    pub fn denormalize(&mut self, mean: &[f32; 3], std: &[f32; 3]) -> Result<(), TensorError> {
        let (c, h, w) = self.spatial_dimensions_3ch()?;
        let plane_size = h * w;

        for ch in 0..c {
            let m = mean[ch];
            let s = std[ch];
            let start = ch * plane_size;
            let end = start + plane_size;
            let slice = &mut self.data[start..end];
            slice.par_iter_mut().for_each(|val| {
                *val = *val * s + m;
            });
        }
        Ok(())
    }

    /// Rescales all elements by a scalar factor.
    pub fn scale(&mut self, factor: f32) {
        self.data.par_iter_mut().for_each(|v| *v *= factor);
    }

    /// Clamps all elements within [min, max].
    pub fn clamp(&mut self, min: f32, max: f32) {
        self.data
            .par_iter_mut()
            .for_each(|v| *v = v.clamp(min, max));
    }

    /// Bilinear resizing of an NCHW tensor to target dimensions (target_h, target_w).
    pub fn resize_bilinear(&self, target_h: usize, target_w: usize) -> Result<Self, TensorError> {
        let (c, src_h, src_w) = match self.shape.as_slice() {
            [1, c, h, w] => (*c, *h, *w),
            [c, h, w] => (*c, *h, *w),
            _ => {
                return Err(TensorError::InvalidDimensions(format!(
                    "Bilinear resize requires 3D or 4D tensor, got {:?}",
                    self.shape
                )))
            }
        };

        if src_h == 0 || src_w == 0 {
            return Err(TensorError::InvalidDimensions(
                "Source dimensions must be greater than zero".to_string(),
            ));
        }

        if target_h == 0 || target_w == 0 {
            return Err(TensorError::InvalidDimensions(
                "Target dimensions must be greater than zero".to_string(),
            ));
        }

        let out_plane_size = target_h * target_w;
        let mut out_data = vec![0.0f32; c * out_plane_size];

        let scale_y = src_h as f32 / target_h as f32;
        let scale_x = src_w as f32 / target_w as f32;

        let src_plane_size = src_h * src_w;

        for ch in 0..c {
            let src_slice = &self.data[ch * src_plane_size..(ch + 1) * src_plane_size];
            let out_slice = &mut out_data[ch * out_plane_size..(ch + 1) * out_plane_size];

            out_slice
                .par_chunks_exact_mut(target_w)
                .enumerate()
                .for_each(|(y_dst, row)| {
                    let y_src_f = (y_dst as f32 + 0.5) * scale_y - 0.5;
                    let y0 = (y_src_f.floor().max(0.0) as usize).min(src_h - 1);
                    let y1 = (y0 + 1).min(src_h - 1);
                    let fy = (y_src_f - y0 as f32).clamp(0.0, 1.0);

                    for (x_dst, out_val) in row.iter_mut().enumerate() {
                        let x_src_f = (x_dst as f32 + 0.5) * scale_x - 0.5;
                        let x0 = (x_src_f.floor().max(0.0) as usize).min(src_w - 1);
                        let x1 = (x0 + 1).min(src_w - 1);
                        let fx = (x_src_f - x0 as f32).clamp(0.0, 1.0);

                        let p00 = src_slice[y0 * src_w + x0];
                        let p01 = src_slice[y0 * src_w + x1];
                        let p10 = src_slice[y1 * src_w + x0];
                        let p11 = src_slice[y1 * src_w + x1];

                        let top = p00 * (1.0 - fx) + p01 * fx;
                        let bottom = p10 * (1.0 - fx) + p11 * fx;
                        *out_val = top * (1.0 - fy) + bottom * fy;
                    }
                });
        }

        let out_shape = if self.shape.len() == 4 {
            vec![1, c, target_h, target_w]
        } else {
            vec![c, target_h, target_w]
        };

        Ok(Self {
            shape: out_shape,
            data: out_data,
        })
    }

    /// Internal helper to extract (channels, height, width) for 3-channel tensors.
    fn spatial_dimensions_3ch(&self) -> Result<(usize, usize, usize), TensorError> {
        match self.shape.as_slice() {
            [1, c, h, w] if *c == 3 => Ok((*c, *h, *w)),
            [c, h, w] if *c == 3 => Ok((*c, *h, *w)),
            _ => Err(TensorError::InvalidDimensions(format!(
                "Expected tensor with 3 channels, got shape {:?}",
                self.shape
            ))),
        }
    }
}
