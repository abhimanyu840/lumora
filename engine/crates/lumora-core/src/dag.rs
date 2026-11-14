use crate::buffer::PixelBufferF32;
use crate::color::{linear_to_srgb, srgb_to_linear};
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};

const CHUNK_PIXELS: usize = 1024;
const CHUNK_FLOATS: usize = CHUNK_PIXELS * 4;

static NEXT_NODE_ID: AtomicU64 = AtomicU64::new(1);

pub trait ImageOp: Send + Sync {
    fn name(&self) -> &str;
    fn apply(&self, buffer: &mut PixelBufferF32);

    /// Apply the operation to an extracted tile with awareness of its global image coordinates.
    ///
    /// Point-wise operations default to invoking `self.apply(tile)`. Spatial operations
    /// (e.g. U-Point control pins) override this to preserve exact coordinates without distortion.
    fn apply_tile(
        &self,
        tile: &mut PixelBufferF32,
        _origin_x: usize,
        _origin_y: usize,
        _full_width: usize,
        _full_height: usize,
    ) {
        self.apply(tile);
    }
}

// Built-in Core Image Operations

#[derive(Debug, Clone)]
pub struct ExposureOp {
    pub ev: f32, // Exposure value in stops (-5.0 to +5.0)
}

impl ExposureOp {
    pub fn new(ev: f32) -> Self {
        Self { ev }
    }
}

impl ImageOp for ExposureOp {
    fn name(&self) -> &str {
        "Exposure"
    }

    fn apply(&self, buffer: &mut PixelBufferF32) {
        if self.ev.abs() < 1e-5 {
            return;
        }
        let mult = 2.0f32.powf(self.ev);
        // Precompute 2048-entry linear-to-sRGB exposure LUT for fast evaluation
        const LUT_SIZE: usize = 2048;
        let mut lut = [0.0f32; LUT_SIZE];
        let max_idx = (LUT_SIZE - 1) as f32;
        for (i, val) in lut.iter_mut().enumerate() {
            let srgb_val = i as f32 / max_idx;
            let lin_val = srgb_to_linear(srgb_val) * mult;
            *val = linear_to_srgb(lin_val).clamp(0.0, 1.0);
        }

        // Process in parallel using chunked linear color space
        buffer.data.par_chunks_mut(CHUNK_FLOATS).for_each(|slice| {
            for chunk in slice.chunks_exact_mut(4) {
                let eval = |c: f32| -> f32 {
                    let pos = (c.clamp(0.0, 1.0) * max_idx).clamp(0.0, max_idx);
                    let i0 = pos as usize;
                    let i1 = (i0 + 1).min(LUT_SIZE - 1);
                    let frac = pos - i0 as f32;
                    lut[i0] * (1.0 - frac) + lut[i1] * frac
                };
                chunk[0] = eval(chunk[0]);
                chunk[1] = eval(chunk[1]);
                chunk[2] = eval(chunk[2]);
            }
        });
    }
}

#[derive(Debug, Clone)]
pub struct ContrastOp {
    pub amount: f32, // -1.0 to 1.0 (0.0 = no-op)
}

impl ContrastOp {
    pub fn new(amount: f32) -> Self {
        Self { amount }
    }
}

impl ImageOp for ContrastOp {
    fn name(&self) -> &str {
        "Contrast"
    }

    fn apply(&self, buffer: &mut PixelBufferF32) {
        if self.amount.abs() < 1e-5 {
            return;
        }
        let factor = if self.amount >= 0.0 {
            1.0 + self.amount * 2.0
        } else {
            1.0 + self.amount * 0.8
        };

        buffer.data.par_chunks_mut(CHUNK_FLOATS).for_each(|slice| {
            for chunk in slice.chunks_exact_mut(4) {
                chunk[0] = ((chunk[0] - 0.5) * factor + 0.5).clamp(0.0, 1.0);
                chunk[1] = ((chunk[1] - 0.5) * factor + 0.5).clamp(0.0, 1.0);
                chunk[2] = ((chunk[2] - 0.5) * factor + 0.5).clamp(0.0, 1.0);
            }
        });
    }
}

#[derive(Debug, Clone)]
pub struct SaturationOp {
    pub amount: f32, // -1.0 (grayscale) to +1.0 (double saturation)
}

impl SaturationOp {
    pub fn new(amount: f32) -> Self {
        Self { amount }
    }
}

impl ImageOp for SaturationOp {
    fn name(&self) -> &str {
        "Saturation"
    }

    fn apply(&self, buffer: &mut PixelBufferF32) {
        if self.amount.abs() < 1e-5 {
            return;
        }
        let mult = 1.0 + self.amount;
        buffer.data.par_chunks_mut(CHUNK_FLOATS).for_each(|slice| {
            for chunk in slice.chunks_exact_mut(4) {
                let lum = 0.2126 * chunk[0] + 0.7152 * chunk[1] + 0.0722 * chunk[2];
                chunk[0] = (lum + (chunk[0] - lum) * mult).clamp(0.0, 1.0);
                chunk[1] = (lum + (chunk[1] - lum) * mult).clamp(0.0, 1.0);
                chunk[2] = (lum + (chunk[2] - lum) * mult).clamp(0.0, 1.0);
            }
        });
    }
}

pub struct DagNode {
    pub id: u64,
    pub name: String,
    pub enabled: bool,
    pub op: Box<dyn ImageOp>,
    pub cached_output: Option<PixelBufferF32>,
}

pub struct PipelineDAG {
    nodes: Vec<DagNode>,
}

impl Default for PipelineDAG {
    fn default() -> Self {
        Self::new()
    }
}

impl PipelineDAG {
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn nodes(&self) -> &[DagNode] {
        &self.nodes
    }

    pub fn node_ids(&self) -> Vec<u64> {
        self.nodes.iter().map(|n| n.id).collect()
    }

    pub fn get_node(&self, id: u64) -> Option<&DagNode> {
        self.nodes.iter().find(|n| n.id == id)
    }

    pub fn update_node<O: ImageOp + 'static>(&mut self, id: u64, op: O) -> bool {
        if let Some(pos) = self.nodes.iter().position(|n| n.id == id) {
            self.nodes[pos].name = op.name().to_string();
            self.nodes[pos].op = Box::new(op);
            self.invalidate_cache_from(pos);
            true
        } else {
            false
        }
    }

    pub fn add_node<O: ImageOp + 'static>(&mut self, op: O) -> u64 {
        let id = NEXT_NODE_ID.fetch_add(1, Ordering::Relaxed);
        let name = op.name().to_string();
        self.nodes.push(DagNode {
            id,
            name,
            enabled: true,
            op: Box::new(op),
            cached_output: None,
        });
        id
    }

    pub fn remove_node(&mut self, id: u64) -> bool {
        if let Some(pos) = self.nodes.iter().position(|n| n.id == id) {
            self.nodes.remove(pos);
            self.invalidate_cache_from(pos);
            true
        } else {
            false
        }
    }

    pub fn set_enabled(&mut self, id: u64, enabled: bool) -> bool {
        if let Some(pos) = self.nodes.iter().position(|n| n.id == id) {
            if self.nodes[pos].enabled != enabled {
                self.nodes[pos].enabled = enabled;
                self.invalidate_cache_from(pos);
            }
            true
        } else {
            false
        }
    }

    pub fn invalidate_cache_from(&mut self, start_idx: usize) {
        for node in &mut self.nodes[start_idx..] {
            node.cached_output = None;
        }
    }

    pub fn clear_cache(&mut self) {
        for node in &mut self.nodes {
            node.cached_output = None;
        }
    }

    /// Execute the full pipeline non-destructively starting from the source buffer.
    pub fn render(&mut self, source: &PixelBufferF32) -> PixelBufferF32 {
        let mut current = source.clone();
        for node in &mut self.nodes {
            if node.enabled {
                node.op.apply(&mut current);
                node.cached_output = Some(current.clone());
            }
        }
        current
    }

    /// Optimized rendering leveraging cached intermediate nodes where possible.
    pub fn render_cached(&mut self, source: &PixelBufferF32) -> PixelBufferF32 {
        // Find the latest valid cached output before any invalidated node
        let mut start_idx = 0;
        let mut current = source.clone();

        for (i, node) in self.nodes.iter().enumerate() {
            if node.enabled {
                if let Some(cached) = &node.cached_output {
                    current = cached.clone();
                    start_idx = i + 1;
                } else {
                    break;
                }
            }
        }

        for node in &mut self.nodes[start_idx..] {
            if node.enabled {
                node.op.apply(&mut current);
                node.cached_output = Some(current.clone());
            }
        }
        current
    }
}
