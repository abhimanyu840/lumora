use crate::cache::AiRasterCache;
use crate::models::{DnCnnModel, MidasModel, RealEsrganModel, ZeroDceModel};
use lumora_core::buffer::PixelBufferF32;
use lumora_core::dag::ImageOp;
use std::sync::{Arc, RwLock};

/// Helper to generate a cache key from model name, buffer dimension, and parameters.
fn make_cache_key(model: &str, w: usize, h: usize, params: &str) -> String {
    format!("{model}_{w}x{h}_{params}")
}

/// Non-destructive AI Denoise DAG Node.
#[derive(Clone)]
pub struct AiDenoiseOp {
    pub strength: f32,
    pub cache: Option<Arc<RwLock<AiRasterCache>>>,
}

impl AiDenoiseOp {
    pub fn new(strength: f32) -> Self {
        Self {
            strength,
            cache: None,
        }
    }

    pub fn with_cache(strength: f32, cache: Arc<RwLock<AiRasterCache>>) -> Self {
        Self {
            strength,
            cache: Some(cache),
        }
    }
}

impl ImageOp for AiDenoiseOp {
    fn name(&self) -> &str {
        "AI Denoise (DnCNN)"
    }

    fn apply(&self, buffer: &mut PixelBufferF32) {
        if self.strength <= 1e-5 {
            return;
        }

        let key = make_cache_key(
            "dncnn",
            buffer.width,
            buffer.height,
            &format!("{:.2}", self.strength),
        );

        if let Some(cache_arc) = &self.cache {
            if let Ok(mut cache) = cache_arc.write() {
                if let Some(cached) = cache.get(&key) {
                    *buffer = cached;
                    return;
                }
            }
        }

        let model = DnCnnModel::new();
        let denoised = model.denoise(buffer, self.strength);

        if let Some(cache_arc) = &self.cache {
            if let Ok(mut cache) = cache_arc.write() {
                cache.insert(key, denoised.clone());
            }
        }

        *buffer = denoised;
    }
}

/// Non-destructive AI Portrait Depth Bokeh DAG Node.
#[derive(Clone)]
pub struct AiDepthBokehOp {
    pub focal_plane: f32,
    pub aperture: f32,
    pub max_blur_radius: f32,
    pub cache: Option<Arc<RwLock<AiRasterCache>>>,
}

impl AiDepthBokehOp {
    pub fn new(focal_plane: f32, aperture: f32, max_blur_radius: f32) -> Self {
        Self {
            focal_plane,
            aperture,
            max_blur_radius,
            cache: None,
        }
    }

    pub fn with_cache(
        focal_plane: f32,
        aperture: f32,
        max_blur_radius: f32,
        cache: Arc<RwLock<AiRasterCache>>,
    ) -> Self {
        Self {
            focal_plane,
            aperture,
            max_blur_radius,
            cache: Some(cache),
        }
    }
}

impl ImageOp for AiDepthBokehOp {
    fn name(&self) -> &str {
        "AI Depth Bokeh (MiDaS)"
    }

    fn apply(&self, buffer: &mut PixelBufferF32) {
        if self.aperture <= 1e-4 || self.max_blur_radius <= 0.5 {
            return;
        }

        let params = format!(
            "{:.2}_{:.2}_{:.1}",
            self.focal_plane, self.aperture, self.max_blur_radius
        );
        let key = make_cache_key("midas_bokeh", buffer.width, buffer.height, &params);

        if let Some(cache_arc) = &self.cache {
            if let Ok(mut cache) = cache_arc.write() {
                if let Some(cached) = cache.get(&key) {
                    *buffer = cached;
                    return;
                }
            }
        }

        let model = MidasModel::new();
        let depth = model.estimate_depth(buffer);
        let bokeh = model.simulate_bokeh(
            buffer,
            &depth,
            self.focal_plane,
            self.aperture,
            self.max_blur_radius,
        );

        if let Some(cache_arc) = &self.cache {
            if let Ok(mut cache) = cache_arc.write() {
                cache.insert(key, bokeh.clone());
            }
        }

        *buffer = bokeh;
    }
}

/// Non-destructive AI 4x Super-Resolution DAG Node.
#[derive(Clone, Default)]
pub struct AiSuperResOp {
    pub cache: Option<Arc<RwLock<AiRasterCache>>>,
}

impl AiSuperResOp {
    pub fn new() -> Self {
        Self { cache: None }
    }

    pub fn with_cache(cache: Arc<RwLock<AiRasterCache>>) -> Self {
        Self { cache: Some(cache) }
    }
}

impl ImageOp for AiSuperResOp {
    fn name(&self) -> &str {
        "AI Super-Resolution 4x (Real-ESRGAN)"
    }

    fn apply(&self, buffer: &mut PixelBufferF32) {
        let key = make_cache_key("realesrgan_4x", buffer.width, buffer.height, "default");

        if let Some(cache_arc) = &self.cache {
            if let Ok(mut cache) = cache_arc.write() {
                if let Some(cached) = cache.get(&key) {
                    *buffer = cached;
                    return;
                }
            }
        }

        let model = RealEsrganModel::new();
        let upscaled = model.upscale_4x(buffer);

        if let Some(cache_arc) = &self.cache {
            if let Ok(mut cache) = cache_arc.write() {
                cache.insert(key, upscaled.clone());
            }
        }

        *buffer = upscaled;
    }
}

/// Non-destructive AI Low-Light Enhancement (Zero-DCE) DAG Node.
#[derive(Clone)]
pub struct AiZeroDceOp {
    pub intensity: f32,
    pub cache: Option<Arc<RwLock<AiRasterCache>>>,
}

impl AiZeroDceOp {
    pub fn new(intensity: f32) -> Self {
        Self {
            intensity,
            cache: None,
        }
    }

    pub fn with_cache(intensity: f32, cache: Arc<RwLock<AiRasterCache>>) -> Self {
        Self {
            intensity,
            cache: Some(cache),
        }
    }
}

impl ImageOp for AiZeroDceOp {
    fn name(&self) -> &str {
        "AI Low-Light Enhancement (Zero-DCE)"
    }

    fn apply(&self, buffer: &mut PixelBufferF32) {
        if self.intensity <= 1e-5 {
            return;
        }

        let key = make_cache_key(
            "zerodce",
            buffer.width,
            buffer.height,
            &format!("{:.2}", self.intensity),
        );

        if let Some(cache_arc) = &self.cache {
            if let Ok(mut cache) = cache_arc.write() {
                if let Some(cached) = cache.get(&key) {
                    *buffer = cached;
                    return;
                }
            }
        }

        let model = ZeroDceModel::new();
        let enhanced = model.enhance(buffer, self.intensity);

        if let Some(cache_arc) = &self.cache {
            if let Ok(mut cache) = cache_arc.write() {
                cache.insert(key, enhanced.clone());
            }
        }

        *buffer = enhanced;
    }
}
