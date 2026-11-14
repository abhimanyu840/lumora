pub mod dncnn;
pub mod midas;
pub mod scene;
pub mod super_res;
pub mod zerodce;

pub use dncnn::DnCnnModel;
pub use midas::{DepthMap, MidasModel};
pub use scene::{AutoEnhancePreset, SceneCategory, SceneClassifier, ScenePrediction};
pub use super_res::RealEsrganModel;
pub use zerodce::ZeroDceModel;
