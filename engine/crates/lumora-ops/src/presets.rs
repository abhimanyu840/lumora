use crate::color_grading::ColorGrading;
use crate::curves::{CurvePoint, ToneCurves};
use crate::hsl::{ChannelAdjust, HslMixer};
use lumora_core::buffer::PixelBufferF32;
use lumora_core::dag::{ContrastOp, ExposureOp, ImageOp, SaturationOp};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

const CHUNK_FLOATS: usize = 1024 * 4;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PresetError {
    InvalidFormat(String),
    ParseError(String),
    UnsupportedVersion(String),
}

impl std::fmt::Display for PresetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidFormat(msg) => write!(f, "Invalid preset format: {msg}"),
            Self::ParseError(msg) => write!(f, "Parse error: {msg}"),
            Self::UnsupportedVersion(msg) => write!(f, "Unsupported version: {msg}"),
        }
    }
}

impl std::error::Error for PresetError {}

/// Native open JSON .lumora preset format
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LumoraPreset {
    pub name: String,
    pub author: String,
    pub version: String,
    pub exposure: Option<f32>,
    pub contrast: Option<f32>,
    pub saturation: Option<f32>,
    pub curves: Option<ToneCurves>,
    pub hsl: Option<HslMixer>,
    pub color_grading: Option<ColorGrading>,
}

impl LumoraPreset {
    pub fn new(name: &str, author: &str) -> Self {
        Self {
            name: name.to_string(),
            author: author.to_string(),
            version: "1.0.0".to_string(),
            exposure: None,
            contrast: None,
            saturation: None,
            curves: None,
            hsl: None,
            color_grading: None,
        }
    }

    pub fn to_json(&self) -> Result<String, PresetError> {
        serde_json::to_string_pretty(self)
            .map_err(|e| PresetError::ParseError(format!("JSON serialization failed: {e}")))
    }

    pub fn from_json(json_str: &str) -> Result<Self, PresetError> {
        serde_json::from_str(json_str)
            .map_err(|e| PresetError::ParseError(format!("JSON deserialization failed: {e}")))
    }

    /// Apply all preset adjustments sequentially to a buffer
    pub fn apply_to_buffer(&self, buffer: &mut PixelBufferF32) {
        if let Some(ev) = self.exposure {
            ExposureOp::new(ev).apply(buffer);
        }
        if let Some(cont) = self.contrast {
            ContrastOp::new(cont).apply(buffer);
        }
        if let Some(sat) = self.saturation {
            SaturationOp::new(sat).apply(buffer);
        }
        if let Some(curves) = &self.curves {
            curves.apply_to_buffer(buffer);
        }
        if let Some(hsl) = &self.hsl {
            hsl.apply_to_buffer(buffer);
        }
        if let Some(grading) = &self.color_grading {
            grading.apply_to_buffer(buffer);
        }
    }
}

/// 3D LUT (.cube format) parser and trilinear evaluator
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CubeLut {
    pub title: String,
    pub size: usize,
    pub domain_min: [f32; 3],
    pub domain_max: [f32; 3],
    pub data: Vec<[f32; 3]>,
}

impl CubeLut {
    pub fn parse(content: &str) -> Result<Self, PresetError> {
        let mut title = "Untitled LUT".to_string();
        let mut size = 0usize;
        let mut domain_min = [0.0f32, 0.0, 0.0];
        let mut domain_max = [1.0f32, 1.0, 1.0];
        let mut table = Vec::new();

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            if line.starts_with("TITLE") {
                let parts: Vec<&str> = line.splitn(2, ' ').collect();
                if parts.len() > 1 {
                    title = parts[1].trim_matches('"').to_string();
                }
            } else if line.starts_with("LUT_3D_SIZE") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() > 1 {
                    size = parts[1]
                        .parse::<usize>()
                        .map_err(|e| PresetError::ParseError(format!("Invalid LUT size: {e}")))?;
                }
            } else if line.starts_with("DOMAIN_MIN") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() == 4 {
                    domain_min[0] = parts[1].parse().unwrap_or(0.0);
                    domain_min[1] = parts[2].parse().unwrap_or(0.0);
                    domain_min[2] = parts[3].parse().unwrap_or(0.0);
                }
            } else if line.starts_with("DOMAIN_MAX") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() == 4 {
                    domain_max[0] = parts[1].parse().unwrap_or(1.0);
                    domain_max[1] = parts[2].parse().unwrap_or(1.0);
                    domain_max[2] = parts[3].parse().unwrap_or(1.0);
                }
            } else {
                let tokens: Vec<&str> = line.split_whitespace().collect();
                if tokens.len() == 3 {
                    if let (Ok(r), Ok(g), Ok(b)) = (
                        tokens[0].parse::<f32>(),
                        tokens[1].parse::<f32>(),
                        tokens[2].parse::<f32>(),
                    ) {
                        table.push([r, g, b]);
                    }
                }
            }
        }

        if size == 0 {
            return Err(PresetError::InvalidFormat(
                "Missing LUT_3D_SIZE parameter".to_string(),
            ));
        }

        let expected_count = size * size * size;
        if table.len() != expected_count {
            return Err(PresetError::InvalidFormat(format!(
                "Incomplete LUT table: expected {expected_count} entries, got {}",
                table.len()
            )));
        }

        Ok(Self {
            title,
            size,
            domain_min,
            domain_max,
            data: table,
        })
    }

    #[inline(always)]
    pub fn sample_trilinear(&self, r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        let size = self.size;
        let s_f = (size - 1) as f32;

        let nr = ((r - self.domain_min[0]) / (self.domain_max[0] - self.domain_min[0]))
            .clamp(0.0, 1.0)
            * s_f;
        let ng = ((g - self.domain_min[1]) / (self.domain_max[1] - self.domain_min[1]))
            .clamp(0.0, 1.0)
            * s_f;
        let nb = ((b - self.domain_min[2]) / (self.domain_max[2] - self.domain_min[2]))
            .clamp(0.0, 1.0)
            * s_f;

        let r0 = (nr as usize).min(size - 1);
        let g0 = (ng as usize).min(size - 1);
        let b0 = (nb as usize).min(size - 1);

        let r1 = (r0 + 1).min(size - 1);
        let g1 = (g0 + 1).min(size - 1);
        let b1 = (b0 + 1).min(size - 1);

        let fr = nr - r0 as f32;
        let fg = ng - g0 as f32;
        let fb = nb - b0 as f32;

        let idx = |r: usize, g: usize, b: usize| -> usize { r + g * size + b * size * size };

        let c000 = self.data[idx(r0, g0, b0)];
        let c100 = self.data[idx(r1, g0, b0)];
        let c010 = self.data[idx(r0, g1, b0)];
        let c110 = self.data[idx(r1, g1, b0)];
        let c001 = self.data[idx(r0, g0, b1)];
        let c101 = self.data[idx(r1, g0, b1)];
        let c011 = self.data[idx(r0, g1, b1)];
        let c111 = self.data[idx(r1, g1, b1)];

        let mut out = [0.0f32; 3];
        for i in 0..3 {
            let c00 = c000[i] * (1.0 - fr) + c100[i] * fr;
            let c10 = c010[i] * (1.0 - fr) + c110[i] * fr;
            let c01 = c001[i] * (1.0 - fr) + c101[i] * fr;
            let c11 = c011[i] * (1.0 - fr) + c111[i] * fr;

            let c0 = c00 * (1.0 - fg) + c10 * fg;
            let c1 = c01 * (1.0 - fg) + c11 * fg;

            out[i] = c0 * (1.0 - fb) + c1 * fb;
        }

        (
            out[0].clamp(0.0, 1.0),
            out[1].clamp(0.0, 1.0),
            out[2].clamp(0.0, 1.0),
        )
    }

    pub fn apply_to_buffer(&self, buffer: &mut PixelBufferF32) {
        buffer.data.par_chunks_mut(CHUNK_FLOATS).for_each(|slice| {
            for chunk in slice.chunks_exact_mut(4) {
                let (r, g, b) = self.sample_trilinear(chunk[0], chunk[1], chunk[2]);
                chunk[0] = r;
                chunk[1] = g;
                chunk[2] = b;
            }
        });
    }
}

impl ImageOp for CubeLut {
    fn name(&self) -> &str {
        "CubeLut"
    }

    fn apply(&self, buffer: &mut PixelBufferF32) {
        self.apply_to_buffer(buffer);
    }
}

/// Adobe Camera Raw .xmp XML preset parser
#[derive(Debug, Clone, PartialEq, Default)]
pub struct XmpPreset {
    pub exposure: Option<f32>,
    pub contrast: Option<f32>,
    pub saturation: Option<f32>,
    pub tone_curve: Option<Vec<CurvePoint>>,
    pub red_curve: Option<Vec<CurvePoint>>,
    pub green_curve: Option<Vec<CurvePoint>>,
    pub blue_curve: Option<Vec<CurvePoint>>,
    pub hsl: HslMixer,
}

impl XmpPreset {
    pub fn parse_xmp(xml_content: &str) -> Result<Self, PresetError> {
        let mut preset = XmpPreset::default();

        let extract_attr = |key: &str| -> Option<String> {
            let search = format!("{key}=\"");
            if let Some(start) = xml_content.find(&search) {
                let rest = &xml_content[start + search.len()..];
                if let Some(end) = rest.find('"') {
                    return Some(rest[..end].to_string());
                }
            }
            // Also try crs:key="value"
            let search_tag = format!("<crs:{key}>");
            if let Some(start) = xml_content.find(&search_tag) {
                let rest = &xml_content[start + search_tag.len()..];
                let close_tag = format!("</crs:{key}>");
                if let Some(end) = rest.find(&close_tag) {
                    return Some(rest[..end].trim().to_string());
                }
            }
            None
        };

        if let Some(val) = extract_attr("Exposure2012") {
            preset.exposure = val.parse::<f32>().ok();
        }
        if let Some(val) = extract_attr("Contrast2012") {
            preset.contrast = val.parse::<f32>().map(|c| c / 100.0).ok();
        }
        if let Some(val) = extract_attr("Saturation") {
            preset.saturation = val.parse::<f32>().map(|s| s / 100.0).ok();
        }

        // Parse ToneCurve points: format like "0, 0, 64, 50, 192, 210, 255, 255"
        let parse_curve_string = |val_str: &str| -> Option<Vec<CurvePoint>> {
            let clean = val_str.trim_matches(|c| c == '[' || c == ']' || c == '(' || c == ')');
            let nums: Vec<f32> = clean
                .split(',')
                .filter_map(|s| s.trim().parse::<f32>().ok())
                .collect();
            if nums.len() >= 4 && nums.len().is_multiple_of(2) {
                let mut pts = Vec::with_capacity(nums.len() / 2);
                for i in (0..nums.len()).step_by(2) {
                    let x = if nums[i] > 1.0 {
                        nums[i] / 255.0
                    } else {
                        nums[i]
                    };
                    let y = if nums[i + 1] > 1.0 {
                        nums[i + 1] / 255.0
                    } else {
                        nums[i + 1]
                    };
                    pts.push(CurvePoint::new(x, y));
                }
                Some(pts)
            } else {
                None
            }
        };

        if let Some(val) = extract_attr("ToneCurvePV2012") {
            preset.tone_curve = parse_curve_string(&val);
        }
        if let Some(val) = extract_attr("ToneCurvePV2012Red") {
            preset.red_curve = parse_curve_string(&val);
        }
        if let Some(val) = extract_attr("ToneCurvePV2012Green") {
            preset.green_curve = parse_curve_string(&val);
        }
        if let Some(val) = extract_attr("ToneCurvePV2012Blue") {
            preset.blue_curve = parse_curve_string(&val);
        }

        // Parse HSL channel adjustments
        let parse_hsl_channel = |color: &str| -> ChannelAdjust {
            let h = extract_attr(&format!("HueAdjustment{color}"))
                .and_then(|v| v.parse::<f32>().ok())
                .unwrap_or(0.0);
            let s = extract_attr(&format!("SaturationAdjustment{color}"))
                .and_then(|v| v.parse::<f32>().ok())
                .map(|v| v / 100.0)
                .unwrap_or(0.0);
            let l = extract_attr(&format!("LuminanceAdjustment{color}"))
                .and_then(|v| v.parse::<f32>().ok())
                .map(|v| v / 100.0)
                .unwrap_or(0.0);
            ChannelAdjust {
                hue: h,
                saturation: s,
                luminance: l,
            }
        };

        preset.hsl.red = parse_hsl_channel("Red");
        preset.hsl.orange = parse_hsl_channel("Orange");
        preset.hsl.yellow = parse_hsl_channel("Yellow");
        preset.hsl.green = parse_hsl_channel("Green");
        preset.hsl.aqua = parse_hsl_channel("Aqua");
        preset.hsl.blue = parse_hsl_channel("Blue");
        preset.hsl.purple = parse_hsl_channel("Purple");
        preset.hsl.magenta = parse_hsl_channel("Magenta");

        Ok(preset)
    }

    pub fn to_tone_curves(&self) -> ToneCurves {
        let mut curves = ToneCurves::linear();
        if let Some(master) = &self.tone_curve {
            curves.master = master.clone();
        }
        if let Some(red) = &self.red_curve {
            curves.red = red.clone();
        }
        if let Some(green) = &self.green_curve {
            curves.green = green.clone();
        }
        if let Some(blue) = &self.blue_curve {
            curves.blue = blue.clone();
        }
        curves
    }
}
