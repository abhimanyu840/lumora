use lumora_core::buffer::PixelBufferF32;
use lumora_core::dag::ImageOp;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

pub const LUT_SIZE: usize = 1024;
const CHUNK_FLOATS: usize = 1024 * 4;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CurvePoint {
    pub x: f32, // 0.0 to 1.0
    pub y: f32, // 0.0 to 1.0
}

impl CurvePoint {
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

/// Precomputed monotone cubic spline (Fritsch-Carlson)
pub struct MonotoneCubicSpline {
    points: Vec<CurvePoint>,
    m: Vec<f32>,
}

impl MonotoneCubicSpline {
    pub fn from_points(raw_points: &[CurvePoint]) -> Self {
        if raw_points.is_empty() {
            return Self {
                points: vec![CurvePoint::new(0.0, 0.0), CurvePoint::new(1.0, 1.0)],
                m: vec![1.0, 1.0],
            };
        }

        // 1. Sort points by x coordinate
        let mut sorted = raw_points.to_vec();
        sorted.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal));

        // 2. Deduplicate points with identical or near-identical x
        let mut points: Vec<CurvePoint> = Vec::with_capacity(sorted.len());
        for p in sorted {
            if let Some(last) = points.last_mut() {
                if (p.x - last.x).abs() < 1e-6 {
                    // Update y value for duplicate x
                    last.y = p.y;
                    continue;
                }
            }
            points.push(p);
        }

        let n = points.len();
        if n == 1 {
            return Self {
                points,
                m: vec![0.0],
            };
        }

        // 3. Compute secants delta_k
        let mut deltas = Vec::with_capacity(n - 1);
        for k in 0..n - 1 {
            let dx = points[k + 1].x - points[k].x;
            if dx.abs() < 1e-7 {
                deltas.push(0.0);
            } else {
                deltas.push((points[k + 1].y - points[k].y) / dx);
            }
        }

        // 4. Initial tangents
        let mut m = vec![0.0f32; n];
        m[0] = deltas[0];
        m[n - 1] = deltas[n - 2];
        for k in 1..n - 1 {
            m[k] = (deltas[k - 1] + deltas[k]) * 0.5;
        }

        // 5. Fritsch-Carlson monotonicity condition
        for k in 0..n - 1 {
            if deltas[k].abs() < 1e-7 {
                m[k] = 0.0;
                m[k + 1] = 0.0;
            } else {
                let alpha = m[k] / deltas[k];
                let beta = m[k + 1] / deltas[k];
                if alpha < 0.0 {
                    m[k] = 0.0;
                }
                if beta < 0.0 {
                    m[k + 1] = 0.0;
                }
                let hyp = alpha * alpha + beta * beta;
                if hyp > 9.0 {
                    let tau = 3.0 / hyp.sqrt();
                    m[k] = tau * alpha * deltas[k];
                    m[k + 1] = tau * beta * deltas[k];
                }
            }
        }

        Self { points, m }
    }

    #[inline]
    pub fn eval(&self, x: f32) -> f32 {
        let x = x.clamp(0.0, 1.0);
        let n = self.points.len();
        if n == 0 {
            return x;
        }
        if n == 1 {
            return self.points[0].y.clamp(0.0, 1.0);
        }
        if x <= self.points[0].x {
            return self.points[0].y.clamp(0.0, 1.0);
        }
        if x >= self.points[n - 1].x {
            return self.points[n - 1].y.clamp(0.0, 1.0);
        }

        // Binary search for interval [k, k+1]
        let mut low = 0;
        let mut high = n - 1;
        while high - low > 1 {
            let mid = (low + high) / 2;
            if self.points[mid].x <= x {
                low = mid;
            } else {
                high = mid;
            }
        }
        let k = low;

        let h = self.points[k + 1].x - self.points[k].x;
        if h.abs() < 1e-7 {
            return self.points[k].y.clamp(0.0, 1.0);
        }

        let t = (x - self.points[k].x) / h;
        let t2 = t * t;
        let t3 = t2 * t;

        let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
        let h10 = t3 - 2.0 * t2 + t;
        let h01 = -2.0 * t3 + 3.0 * t2;
        let h11 = t3 - t2;

        let y = h00 * self.points[k].y
            + h10 * h * self.m[k]
            + h01 * self.points[k + 1].y
            + h11 * h * self.m[k + 1];
        y.clamp(0.0, 1.0)
    }
}

/// Evaluates a monotone cubic spline (Fritsch-Carlson)
pub fn evaluate_spline(points: &[CurvePoint], x: f32) -> f32 {
    let spline = MonotoneCubicSpline::from_points(points);
    spline.eval(x)
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToneCurves {
    pub master: Vec<CurvePoint>,
    pub red: Vec<CurvePoint>,
    pub green: Vec<CurvePoint>,
    pub blue: Vec<CurvePoint>,
}

impl Default for ToneCurves {
    fn default() -> Self {
        Self::linear()
    }
}

impl ToneCurves {
    pub fn linear() -> Self {
        let default_points = vec![CurvePoint::new(0.0, 0.0), CurvePoint::new(1.0, 1.0)];
        Self {
            master: default_points.clone(),
            red: default_points.clone(),
            green: default_points.clone(),
            blue: default_points,
        }
    }

    pub fn is_identity(&self) -> bool {
        let is_linear = |pts: &[CurvePoint]| -> bool {
            if pts.len() <= 1 {
                return false;
            }
            if pts.len() == 2 {
                return (pts[0].x - 0.0).abs() < 1e-4
                    && (pts[0].y - 0.0).abs() < 1e-4
                    && (pts[1].x - 1.0).abs() < 1e-4
                    && (pts[1].y - 1.0).abs() < 1e-4;
            }
            pts.iter().all(|p| (p.x - p.y).abs() < 1e-4)
        };
        is_linear(&self.master)
            && is_linear(&self.red)
            && is_linear(&self.green)
            && is_linear(&self.blue)
    }

    pub fn build_lut(points: &[CurvePoint]) -> [f32; LUT_SIZE] {
        let spline = MonotoneCubicSpline::from_points(points);
        let mut lut = [0.0f32; LUT_SIZE];
        let max_idx = (LUT_SIZE - 1) as f32;
        for (i, val) in lut.iter_mut().enumerate() {
            let x = i as f32 / max_idx;
            *val = spline.eval(x);
        }
        lut
    }

    #[inline(always)]
    pub fn eval_lut(lut: &[f32; LUT_SIZE], val: f32) -> f32 {
        let max_idx = (LUT_SIZE - 1) as f32;
        let pos = (val.clamp(0.0, 1.0) * max_idx).clamp(0.0, max_idx);
        let i0 = pos as usize;
        let i1 = (i0 + 1).min(LUT_SIZE - 1);
        let frac = pos - i0 as f32;
        lut[i0] * (1.0 - frac) + lut[i1] * frac
    }

    pub fn apply_to_buffer(&self, buffer: &mut PixelBufferF32) {
        if self.is_identity() {
            return;
        }

        let master_lut = Self::build_lut(&self.master);
        let red_lut = Self::build_lut(&self.red);
        let green_lut = Self::build_lut(&self.green);
        let blue_lut = Self::build_lut(&self.blue);

        // Pre-compose channel LUTs with master LUT for single-pass 3-LUT evaluation
        let mut composite_r = [0.0f32; LUT_SIZE];
        let mut composite_g = [0.0f32; LUT_SIZE];
        let mut composite_b = [0.0f32; LUT_SIZE];

        for i in 0..LUT_SIZE {
            composite_r[i] = Self::eval_lut(&master_lut, red_lut[i]);
            composite_g[i] = Self::eval_lut(&master_lut, green_lut[i]);
            composite_b[i] = Self::eval_lut(&master_lut, blue_lut[i]);
        }

        buffer.data.par_chunks_mut(CHUNK_FLOATS).for_each(|slice| {
            for chunk in slice.chunks_exact_mut(4) {
                chunk[0] = Self::eval_lut(&composite_r, chunk[0]);
                chunk[1] = Self::eval_lut(&composite_g, chunk[1]);
                chunk[2] = Self::eval_lut(&composite_b, chunk[2]);
            }
        });
    }
}

impl ImageOp for ToneCurves {
    fn name(&self) -> &str {
        "ToneCurves"
    }

    fn apply(&self, buffer: &mut PixelBufferF32) {
        self.apply_to_buffer(buffer);
    }
}
