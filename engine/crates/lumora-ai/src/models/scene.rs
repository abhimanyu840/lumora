use lumora_core::buffer::PixelBufferF32;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SceneCategory {
    Landscape,
    Portrait,
    Night,
    Architecture,
    Sunset,
    Macro,
}

impl std::fmt::Display for SceneCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Landscape => write!(f, "Landscape"),
            Self::Portrait => write!(f, "Portrait"),
            Self::Night => write!(f, "Night"),
            Self::Architecture => write!(f, "Architecture"),
            Self::Sunset => write!(f, "Sunset"),
            Self::Macro => write!(f, "Macro"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenePrediction {
    pub category: SceneCategory,
    pub confidence: f32,
    pub scores: Vec<(SceneCategory, f32)>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AutoEnhancePreset {
    pub name: String,
    pub exposure: f32,
    pub contrast: f32,
    pub saturation: f32,
    pub temperature: f32,
    pub shadows: f32,
    pub highlights: f32,
    pub clarity: f32,
}

#[derive(Debug, Clone, Default)]
pub struct SceneClassifier;

impl SceneClassifier {
    pub fn new() -> Self {
        Self
    }

    /// Classifies an image into one of 6 primary photographic scene categories.
    pub fn classify(&self, input: &PixelBufferF32) -> ScenePrediction {
        let w = input.width;
        let h = input.height;

        if w == 0 || h == 0 {
            let default_score = 1.0 / 6.0;
            let scores = vec![
                (SceneCategory::Landscape, default_score),
                (SceneCategory::Portrait, default_score),
                (SceneCategory::Night, default_score),
                (SceneCategory::Architecture, default_score),
                (SceneCategory::Sunset, default_score),
                (SceneCategory::Macro, default_score),
            ];
            return ScenePrediction {
                category: SceneCategory::Landscape,
                confidence: default_score,
                scores,
            };
        }

        let total_pixels = (w * h) as f32;

        let mut sum_r = 0.0f32;
        let mut sum_g = 0.0f32;
        let mut sum_b = 0.0f32;
        let mut sum_lum = 0.0f32;
        let mut dark_pixels = 0.0f32;
        let mut sunset_sky_pixels = 0.0f32;
        let mut skin_pixels = 0.0f32;
        let mut foliage_pixels = 0.0f32;
        let mut center_sharpness = 0.0f32;
        let mut border_sharpness = 0.0f32;

        let half_h = h / 2;
        let center_box_x = (w / 4)..((3 * w) / 4);
        let center_box_y = (h / 4)..((3 * h) / 4);

        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) * 4;
                let r = input.data[idx];
                let g = input.data[idx + 1];
                let b = input.data[idx + 2];

                let lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                sum_r += r;
                sum_g += g;
                sum_b += b;
                sum_lum += lum;

                if lum < 0.15 {
                    dark_pixels += 1.0;
                }

                // Sunset detection: high warmth in upper half of frame with high red-to-blue ratio
                if y < half_h && r > 0.5 && r > b * 1.8 {
                    sunset_sky_pixels += 1.0;
                }

                // Foliage / green landscape
                if g > r * 1.1 && g > b * 1.2 {
                    foliage_pixels += 1.0;
                }

                // Portrait skin tone heuristic: moderate saturation, natural skin hue, not pure sky
                let is_skin_tone = r > g && g > b && (r - g) < 0.35 && (g - b) > 0.03 && b > 0.15;
                if is_skin_tone {
                    skin_pixels += 1.0;
                }

                // Local gradient for sharpness
                if x + 1 < w && y + 1 < h {
                    let next_r = input.data[idx + 4];
                    let grad = (r - next_r).abs();
                    if center_box_x.contains(&x) && center_box_y.contains(&y) {
                        center_sharpness += grad;
                    } else {
                        border_sharpness += grad;
                    }
                }
            }
        }

        let mean_lum = sum_lum / total_pixels;
        let dark_ratio = dark_pixels / total_pixels;
        let sunset_ratio = sunset_sky_pixels / (total_pixels * 0.5).max(1.0);
        let foliage_ratio = foliage_pixels / total_pixels;
        let skin_ratio = skin_pixels / total_pixels;
        let macro_ratio = if border_sharpness > 1e-4 {
            (center_sharpness / (border_sharpness + 1.0)).min(3.0) / 3.0
        } else {
            0.0
        };

        // Score per class
        let raw_scores = [
            (
                SceneCategory::Landscape,
                foliage_ratio * 3.5
                    + if sum_g > sum_b && mean_lum > 0.35 {
                        0.4
                    } else {
                        0.0
                    },
            ),
            (SceneCategory::Portrait, skin_ratio * 4.0),
            (
                SceneCategory::Night,
                dark_ratio * 3.5 + if mean_lum < 0.25 { 0.8 } else { 0.0 },
            ),
            (
                SceneCategory::Architecture,
                (border_sharpness / total_pixels) * 50.0,
            ),
            (
                SceneCategory::Sunset,
                sunset_ratio * 4.5 + if sum_r > sum_b * 1.3 { 0.6 } else { 0.0 },
            ),
            (SceneCategory::Macro, macro_ratio * 2.5),
        ];

        // Softmax normalization
        let max_val = raw_scores
            .iter()
            .map(|(_, s)| *s)
            .fold(f32::NEG_INFINITY, f32::max);
        let mut exp_sum = 0.0f32;
        let mut exp_scores: Vec<(SceneCategory, f32)> = raw_scores
            .iter()
            .map(|(cat, s)| {
                let e = (s - max_val).exp();
                exp_sum += e;
                (*cat, e)
            })
            .collect();

        for (_, s) in &mut exp_scores {
            *s = (*s / exp_sum.max(1e-6)).clamp(0.0, 1.0);
        }

        // Sort descending by score
        exp_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top_cat = exp_scores[0].0;
        let confidence = exp_scores[0].1;

        ScenePrediction {
            category: top_cat,
            confidence,
            scores: exp_scores,
        }
    }

    /// Recommends professional adjustment parameters based on detected scene.
    pub fn recommend_preset(&self, prediction: &ScenePrediction) -> AutoEnhancePreset {
        match prediction.category {
            SceneCategory::Landscape => AutoEnhancePreset {
                name: "AI Vibrant Landscape".to_string(),
                exposure: 0.15,
                contrast: 0.20,
                saturation: 0.25,
                temperature: 0.05,
                shadows: 0.20,
                highlights: -0.15,
                clarity: 0.30,
            },
            SceneCategory::Portrait => AutoEnhancePreset {
                name: "AI Portrait Glow".to_string(),
                exposure: 0.20,
                contrast: 0.05,
                saturation: 0.08,
                temperature: 0.10,
                shadows: 0.15,
                highlights: -0.10,
                clarity: -0.05, // Gentle texture softening for skin
            },
            SceneCategory::Night => AutoEnhancePreset {
                name: "AI Night Recovery".to_string(),
                exposure: 0.65,
                contrast: 0.25,
                saturation: 0.10,
                temperature: -0.05,
                shadows: 0.45,
                highlights: -0.30,
                clarity: 0.15,
            },
            SceneCategory::Architecture => AutoEnhancePreset {
                name: "AI Clean Geometry".to_string(),
                exposure: 0.0,
                contrast: 0.30,
                saturation: 0.05,
                temperature: 0.0,
                shadows: 0.15,
                highlights: -0.20,
                clarity: 0.40,
            },
            SceneCategory::Sunset => AutoEnhancePreset {
                name: "AI Golden Hour".to_string(),
                exposure: -0.10,
                contrast: 0.20,
                saturation: 0.35,
                temperature: 0.30,
                shadows: 0.10,
                highlights: -0.25,
                clarity: 0.20,
            },
            SceneCategory::Macro => AutoEnhancePreset {
                name: "AI Macro Detail".to_string(),
                exposure: 0.10,
                contrast: 0.15,
                saturation: 0.20,
                temperature: 0.05,
                shadows: 0.10,
                highlights: -0.10,
                clarity: 0.50,
            },
        }
    }
}
