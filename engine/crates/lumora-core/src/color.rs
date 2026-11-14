use serde::{Deserialize, Serialize};

// D65 Standard Illuminant reference white point
pub const D65_XN: f32 = 0.95047;
pub const D65_YN: f32 = 1.00000;
pub const D65_ZN: f32 = 1.08883;

const DELTA: f32 = 6.0 / 29.0;
const DELTA_CUBED: f32 = DELTA * DELTA * DELTA;
const THREE_DELTA_SQ: f32 = 3.0 * DELTA * DELTA;
const FOUR_OVER_TWENTY_NINE: f32 = 4.0 / 29.0;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Lab {
    pub l: f32, // 0.0 to 100.0
    pub a: f32, // -128.0 to 127.0
    pub b: f32, // -128.0 to 127.0
}

impl Lab {
    #[inline(always)]
    pub const fn new(l: f32, a: f32, b: f32) -> Self {
        Self { l, a, b }
    }

    #[inline(always)]
    pub fn delta_e_cie76(self, other: Lab) -> f32 {
        let dl = self.l - other.l;
        let da = self.a - other.a;
        let db = self.b - other.b;
        (dl * dl + da * da + db * db).sqrt()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Hsl {
    pub h: f32, // 0.0 to 360.0 degrees
    pub s: f32, // 0.0 to 1.0
    pub l: f32, // 0.0 to 1.0
}

impl Hsl {
    #[inline(always)]
    pub const fn new(h: f32, s: f32, l: f32) -> Self {
        Self { h, s, l }
    }
}

/// Convert sRGB channel in [0.0, 1.0] to Linear RGB channel
#[inline(always)]
pub fn srgb_to_linear(c: f32) -> f32 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Convert Linear RGB channel in [0.0, 1.0] to sRGB channel
#[inline(always)]
pub fn linear_to_srgb(c: f32) -> f32 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.0031308 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

/// Convert Linear RGB (D65) to CIE XYZ
#[inline(always)]
pub fn linear_to_xyz(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b;
    let y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b;
    let z = 0.0193339 * r + 0.119192 * g + 0.9503041 * b;
    (x, y, z)
}

/// Convert CIE XYZ to Linear RGB (D65)
#[inline(always)]
pub fn xyz_to_linear(x: f32, y: f32, z: f32) -> (f32, f32, f32) {
    let r = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z;
    let g = -0.969266 * x + 1.8760108 * y + 0.041556 * z;
    let b = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z;
    (r, g, b)
}

#[inline(always)]
fn f_lab(t: f32) -> f32 {
    if t > DELTA_CUBED {
        t.cbrt()
    } else {
        t / THREE_DELTA_SQ + FOUR_OVER_TWENTY_NINE
    }
}

#[inline(always)]
fn f_inv_lab(t: f32) -> f32 {
    if t > DELTA {
        t * t * t
    } else {
        THREE_DELTA_SQ * (t - FOUR_OVER_TWENTY_NINE)
    }
}

/// Convert CIE XYZ (D65) to CIE-Lab
#[inline(always)]
pub fn xyz_to_lab(x: f32, y: f32, z: f32) -> Lab {
    let fx = f_lab(x / D65_XN);
    let fy = f_lab(y / D65_YN);
    let fz = f_lab(z / D65_ZN);

    let l = 116.0 * fy - 16.0;
    let a = 500.0 * (fx - fy);
    let b = 200.0 * (fy - fz);
    Lab { l, a, b }
}

/// Convert CIE-Lab to CIE XYZ (D65)
#[inline(always)]
pub fn lab_to_xyz(lab: Lab) -> (f32, f32, f32) {
    let fy = (lab.l + 16.0) / 116.0;
    let fx = lab.a / 500.0 + fy;
    let fz = fy - lab.b / 200.0;

    let x = D65_XN * f_inv_lab(fx);
    let y = D65_YN * f_inv_lab(fy);
    let z = D65_ZN * f_inv_lab(fz);
    (x, y, z)
}

/// Directly convert sRGB components [0.0, 1.0] to CIE-Lab
#[inline(always)]
pub fn srgb_to_lab(r: f32, g: f32, b: f32) -> Lab {
    let lr = srgb_to_linear(r);
    let lg = srgb_to_linear(g);
    let lb = srgb_to_linear(b);
    let (x, y, z) = linear_to_xyz(lr, lg, lb);
    xyz_to_lab(x, y, z)
}

/// Directly convert CIE-Lab to sRGB components [0.0, 1.0]
#[inline(always)]
pub fn lab_to_srgb(lab: Lab) -> (f32, f32, f32) {
    let (x, y, z) = lab_to_xyz(lab);
    let (lr, lg, lb) = xyz_to_linear(x, y, z);
    (
        linear_to_srgb(lr).clamp(0.0, 1.0),
        linear_to_srgb(lg).clamp(0.0, 1.0),
        linear_to_srgb(lb).clamp(0.0, 1.0),
    )
}

/// Convert sRGB [0.0, 1.0] to HSL (Hue in [0, 360), Saturation in [0, 1], Lightness in [0, 1])
pub fn rgb_to_hsl(r: f32, g: f32, b: f32) -> Hsl {
    let r = r.clamp(0.0, 1.0);
    let g = g.clamp(0.0, 1.0);
    let b = b.clamp(0.0, 1.0);

    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;

    let l = (max + min) / 2.0;

    if delta.abs() < 1e-6 {
        return Hsl { h: 0.0, s: 0.0, l };
    }

    let s = if l > 0.5 {
        delta / (2.0 - max - min)
    } else {
        delta / (max + min)
    };

    let mut h = if (max - r).abs() < 1e-6 {
        (g - b) / delta + (if g < b { 6.0 } else { 0.0 })
    } else if (max - g).abs() < 1e-6 {
        (b - r) / delta + 2.0
    } else {
        (r - g) / delta + 4.0
    };

    h *= 60.0;
    if h < 0.0 {
        h += 360.0;
    }
    if h >= 360.0 {
        h -= 360.0;
    }

    Hsl { h, s, l }
}

#[inline(always)]
fn hue_to_rgb(p: f32, q: f32, mut t: f32) -> f32 {
    if t < 0.0 {
        t += 1.0;
    }
    if t > 1.0 {
        t -= 1.0;
    }
    if t < 1.0 / 6.0 {
        return p + (q - p) * 6.0 * t;
    }
    if t < 1.0 / 2.0 {
        return q;
    }
    if t < 2.0 / 3.0 {
        return p + (q - p) * (2.0 / 3.0 - t) * 6.0;
    }
    p
}

/// Convert HSL (Hue in [0, 360), Saturation in [0, 1], Lightness in [0, 1]) to sRGB [0.0, 1.0]
pub fn hsl_to_rgb(hsl: Hsl) -> (f32, f32, f32) {
    let s = hsl.s.clamp(0.0, 1.0);
    let l = hsl.l.clamp(0.0, 1.0);

    if s.abs() < 1e-6 {
        return (l, l, l);
    }

    let q = if l < 0.5 {
        l * (1.0 + s)
    } else {
        l + s - l * s
    };
    let p = 2.0 * l - q;
    let hk = (hsl.h % 360.0 + 360.0) % 360.0 / 360.0;

    let r = hue_to_rgb(p, q, hk + 1.0 / 3.0);
    let g = hue_to_rgb(p, q, hk);
    let b = hue_to_rgb(p, q, hk - 1.0 / 3.0);

    (r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0))
}
