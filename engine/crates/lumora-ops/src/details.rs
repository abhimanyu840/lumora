use lumora_core::buffer::PixelBufferF32;
use lumora_core::dag::ImageOp;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DetailsAdjustment {
    pub structure: f32,  // -1.0 to 1.0 (Snapseed texture/structure enhancer)
    pub sharpening: f32, // 0.0 to 1.0 (Unsharp mask detail)
}

impl Default for DetailsAdjustment {
    fn default() -> Self {
        Self {
            structure: 0.0,
            sharpening: 0.0,
        }
    }
}

impl DetailsAdjustment {
    pub fn new(structure: f32, sharpening: f32) -> Self {
        Self {
            structure: structure.clamp(-1.0, 1.0),
            sharpening: sharpening.clamp(0.0, 1.0),
        }
    }

    pub fn apply_to_buffer(&self, buffer: &mut PixelBufferF32) {
        if self.structure.abs() < 1e-4 && self.sharpening < 1e-4 {
            return;
        }

        let width = buffer.width;
        let height = buffer.height;
        if width < 3 || height < 3 {
            return;
        }

        let original = buffer.clone();
        let total_sharp = (self.sharpening * 0.8 + self.structure * 0.5).clamp(-1.0, 2.0);

        // Process in parallel row-by-row using Rayon
        buffer
            .data
            .par_chunks_exact_mut(width * 4)
            .enumerate()
            .filter(|(y, _)| *y >= 1 && *y < height - 1)
            .for_each(|(y, row)| {
                for x in 1..width - 1 {
                    for c in 0..3 {
                        let center = original.data[((y * width + x) * 4) + c];
                        let up = original.data[(((y - 1) * width + x) * 4) + c];
                        let down = original.data[(((y + 1) * width + x) * 4) + c];
                        let left = original.data[((y * width + (x - 1)) * 4) + c];
                        let right = original.data[((y * width + (x + 1)) * 4) + c];

                        // Laplacian high-pass
                        let laplacian = 4.0 * center - (up + down + left + right);
                        let enhanced = center + laplacian * total_sharp * 0.25;
                        row[x * 4 + c] = enhanced.clamp(0.0, 1.0);
                    }
                }
            });
    }
}

impl ImageOp for DetailsAdjustment {
    fn name(&self) -> &str {
        "Details"
    }

    fn apply(&self, buffer: &mut PixelBufferF32) {
        self.apply_to_buffer(buffer);
    }
}
