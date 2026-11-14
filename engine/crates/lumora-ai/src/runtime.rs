use crate::models::{
    DnCnnModel, MidasModel, RealEsrganModel, SceneCategory, SceneClassifier, ZeroDceModel,
};
use crate::tensor::{Tensor, TensorError};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ExecutionProvider {
    #[default]
    Cpu,
    Nnapi,  // Android Hardware Acceleration (API 27+)
    CoreML, // iOS Apple Neural Engine
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub name: String,
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    pub quantized: bool,
    pub provider: ExecutionProvider,
}

#[derive(Debug)]
pub enum AiError {
    ModelNotFound(String),
    InferenceFailed(String),
    Tensor(TensorError),
    Io(std::io::Error),
    InvalidConfiguration(String),
}

impl std::fmt::Display for AiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ModelNotFound(p) => write!(f, "AI Model not found at: {p}"),
            Self::InferenceFailed(msg) => write!(f, "AI Inference failed: {msg}"),
            Self::Tensor(e) => write!(f, "Tensor error: {e}"),
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::InvalidConfiguration(msg) => write!(f, "Invalid AI configuration: {msg}"),
        }
    }
}

impl std::error::Error for AiError {}

impl From<TensorError> for AiError {
    fn from(e: TensorError) -> Self {
        Self::Tensor(e)
    }
}

impl From<std::io::Error> for AiError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// Abstract AI inference session trait.
pub trait AiSession: Send + Sync {
    fn name(&self) -> &str;
    fn metadata(&self) -> &ModelMetadata;
    fn run(&self, input: &Tensor) -> Result<Tensor, AiError>;
}

/// Dedicated DnCNN Denoise Session.
pub struct DnCnnSession {
    model: DnCnnModel,
    strength: f32,
    metadata: ModelMetadata,
}

impl DnCnnSession {
    pub fn new(strength: f32, provider: ExecutionProvider) -> Self {
        Self {
            model: DnCnnModel::new(),
            strength: strength.clamp(0.0, 1.0),
            metadata: ModelMetadata {
                name: "dncnn".to_string(),
                input_shape: vec![1, 3, 0, 0],
                output_shape: vec![1, 3, 0, 0],
                quantized: true,
                provider,
            },
        }
    }
}

impl Default for DnCnnSession {
    fn default() -> Self {
        Self::new(0.8, ExecutionProvider::Cpu)
    }
}

impl AiSession for DnCnnSession {
    fn name(&self) -> &str {
        &self.metadata.name
    }

    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn run(&self, input: &Tensor) -> Result<Tensor, AiError> {
        let buf = input.to_pixel_buffer()?;
        let denoised = self.model.denoise(&buf, self.strength);
        Ok(Tensor::from_pixel_buffer(&denoised))
    }
}

/// Dedicated MiDaS Monocular Depth Estimation Session.
pub struct MidasSession {
    model: MidasModel,
    metadata: ModelMetadata,
}

impl MidasSession {
    pub fn new(provider: ExecutionProvider) -> Self {
        Self {
            model: MidasModel::new(),
            metadata: ModelMetadata {
                name: "midas".to_string(),
                input_shape: vec![1, 3, 0, 0],
                output_shape: vec![1, 1, 0, 0],
                quantized: true,
                provider,
            },
        }
    }
}

impl Default for MidasSession {
    fn default() -> Self {
        Self::new(ExecutionProvider::Cpu)
    }
}

impl AiSession for MidasSession {
    fn name(&self) -> &str {
        &self.metadata.name
    }

    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn run(&self, input: &Tensor) -> Result<Tensor, AiError> {
        let buf = input.to_pixel_buffer()?;
        let depth = self.model.estimate_depth(&buf);
        Tensor::from_data(vec![1, 1, depth.height, depth.width], depth.data).map_err(AiError::from)
    }
}

/// Dedicated MobileNet Scene Classification Session.
pub struct SceneSession {
    classifier: SceneClassifier,
    metadata: ModelMetadata,
}

impl SceneSession {
    pub fn new(provider: ExecutionProvider) -> Self {
        Self {
            classifier: SceneClassifier::new(),
            metadata: ModelMetadata {
                name: "mobilenet_scene".to_string(),
                input_shape: vec![1, 3, 224, 224],
                output_shape: vec![1, 6],
                quantized: true,
                provider,
            },
        }
    }
}

impl Default for SceneSession {
    fn default() -> Self {
        Self::new(ExecutionProvider::Cpu)
    }
}

impl AiSession for SceneSession {
    fn name(&self) -> &str {
        &self.metadata.name
    }

    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn run(&self, input: &Tensor) -> Result<Tensor, AiError> {
        let buf = input.to_pixel_buffer()?;
        let prediction = self.classifier.classify(&buf);
        let category_order = [
            SceneCategory::Landscape,
            SceneCategory::Portrait,
            SceneCategory::Night,
            SceneCategory::Architecture,
            SceneCategory::Sunset,
            SceneCategory::Macro,
        ];

        let mut ordered_scores = Vec::with_capacity(6);
        for cat in &category_order {
            let score = prediction
                .scores
                .iter()
                .find(|(c, _)| c == cat)
                .map(|(_, s)| *s)
                .unwrap_or(0.0);
            ordered_scores.push(score);
        }

        Tensor::from_data(vec![1, 6], ordered_scores).map_err(AiError::from)
    }
}

/// Dedicated Real-ESRGAN 4x Super-Resolution Session.
pub struct SuperResSession {
    model: RealEsrganModel,
    metadata: ModelMetadata,
}

impl SuperResSession {
    pub fn new(provider: ExecutionProvider) -> Self {
        Self {
            model: RealEsrganModel::new(),
            metadata: ModelMetadata {
                name: "realesrgan_x4".to_string(),
                input_shape: vec![1, 3, 0, 0],
                output_shape: vec![1, 3, 0, 0],
                quantized: true,
                provider,
            },
        }
    }
}

impl Default for SuperResSession {
    fn default() -> Self {
        Self::new(ExecutionProvider::Cpu)
    }
}

impl AiSession for SuperResSession {
    fn name(&self) -> &str {
        &self.metadata.name
    }

    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn run(&self, input: &Tensor) -> Result<Tensor, AiError> {
        let buf = input.to_pixel_buffer()?;
        let upscaled = self.model.upscale_4x(&buf);
        Ok(Tensor::from_pixel_buffer(&upscaled))
    }
}

/// Dedicated Zero-DCE Low-Light Enhancement Session.
pub struct ZeroDceSession {
    model: ZeroDceModel,
    intensity: f32,
    metadata: ModelMetadata,
}

impl ZeroDceSession {
    pub fn new(intensity: f32, provider: ExecutionProvider) -> Self {
        Self {
            model: ZeroDceModel::new(),
            intensity: intensity.clamp(0.0, 2.5),
            metadata: ModelMetadata {
                name: "zerodce".to_string(),
                input_shape: vec![1, 3, 0, 0],
                output_shape: vec![1, 3, 0, 0],
                quantized: true,
                provider,
            },
        }
    }
}

impl Default for ZeroDceSession {
    fn default() -> Self {
        Self::new(1.0, ExecutionProvider::Cpu)
    }
}

impl AiSession for ZeroDceSession {
    fn name(&self) -> &str {
        &self.metadata.name
    }

    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn run(&self, input: &Tensor) -> Result<Tensor, AiError> {
        let buf = input.to_pixel_buffer()?;
        let enhanced = self.model.enhance(&buf, self.intensity);
        Ok(Tensor::from_pixel_buffer(&enhanced))
    }
}

/// Runtime engine manager capable of discovering and loading model packs on device.
pub struct AiRuntimeManager {
    models_dir: PathBuf,
    provider: ExecutionProvider,
}

impl AiRuntimeManager {
    pub fn new<P: AsRef<Path>>(models_dir: P, provider: ExecutionProvider) -> Self {
        Self {
            models_dir: models_dir.as_ref().to_path_buf(),
            provider,
        }
    }

    pub fn models_dir(&self) -> &Path {
        &self.models_dir
    }

    pub fn provider(&self) -> ExecutionProvider {
        self.provider
    }

    pub fn model_path(&self, model_filename: &str) -> PathBuf {
        self.models_dir.join(model_filename)
    }

    pub fn is_model_installed(&self, model_filename: &str) -> bool {
        self.model_path(model_filename).exists()
    }

    /// Creates and configures an execution session for the requested model.
    pub fn create_session(&self, model_name: &str) -> Result<Box<dyn AiSession>, AiError> {
        let name_lower = model_name.to_lowercase();
        if name_lower.contains("dncnn") {
            Ok(Box::new(DnCnnSession::new(0.8, self.provider)))
        } else if name_lower.contains("midas") {
            Ok(Box::new(MidasSession::new(self.provider)))
        } else if name_lower.contains("scene") || name_lower.contains("mobilenet") {
            Ok(Box::new(SceneSession::new(self.provider)))
        } else if name_lower.contains("esrgan") || name_lower.contains("super_res") {
            Ok(Box::new(SuperResSession::new(self.provider)))
        } else if name_lower.contains("zerodce") || name_lower.contains("zero_dce") {
            Ok(Box::new(ZeroDceSession::new(1.0, self.provider)))
        } else {
            Err(AiError::ModelNotFound(format!(
                "Unknown or unsupported model name: {model_name}"
            )))
        }
    }
}
