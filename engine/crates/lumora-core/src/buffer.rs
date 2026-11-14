use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rgba8 {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Rgba8 {
    #[inline(always)]
    pub const fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    #[inline(always)]
    pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b, a: 255 }
    }

    #[inline(always)]
    pub fn to_f32(self) -> RgbaF32 {
        RgbaF32 {
            r: self.r as f32 / 255.0,
            g: self.g as f32 / 255.0,
            b: self.b as f32 / 255.0,
            a: self.a as f32 / 255.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RgbaF32 {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl RgbaF32 {
    #[inline(always)]
    pub const fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    #[inline(always)]
    pub const fn rgb(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b, a: 1.0 }
    }

    #[inline(always)]
    pub fn to_rgba8(self) -> Rgba8 {
        let clamp_byte = |v: f32| -> u8 { (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8 };
        Rgba8 {
            r: clamp_byte(self.r),
            g: clamp_byte(self.g),
            b: clamp_byte(self.b),
            a: clamp_byte(self.a),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BufferError {
    DimensionMismatch {
        expected: usize,
        actual: usize,
    },
    OutOfBounds {
        x: usize,
        y: usize,
        width: usize,
        height: usize,
    },
}

impl std::fmt::Display for BufferError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DimensionMismatch { expected, actual } => {
                write!(
                    f,
                    "Buffer size mismatch: expected {expected} bytes, got {actual}"
                )
            }
            Self::OutOfBounds {
                x,
                y,
                width,
                height,
            } => {
                write!(
                    f,
                    "Pixel coordinates ({x}, {y}) out of bounds for {width}x{height}"
                )
            }
        }
    }
}

impl std::error::Error for BufferError {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PixelBufferRgba8 {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

impl PixelBufferRgba8 {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            data: vec![0u8; width * height * 4],
        }
    }

    pub fn from_raw(width: usize, height: usize, data: Vec<u8>) -> Result<Self, BufferError> {
        let expected = width * height * 4;
        if data.len() != expected {
            return Err(BufferError::DimensionMismatch {
                expected,
                actual: data.len(),
            });
        }
        Ok(Self {
            width,
            height,
            data,
        })
    }

    #[inline(always)]
    pub fn dimensions(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    #[inline(always)]
    pub fn pixel_count(&self) -> usize {
        self.width * self.height
    }

    #[inline(always)]
    pub fn get_pixel(&self, x: usize, y: usize) -> Option<Rgba8> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let idx = (y * self.width + x) * 4;
        Some(Rgba8 {
            r: self.data[idx],
            g: self.data[idx + 1],
            b: self.data[idx + 2],
            a: self.data[idx + 3],
        })
    }

    #[inline(always)]
    pub fn set_pixel(&mut self, x: usize, y: usize, p: Rgba8) -> Result<(), BufferError> {
        if x >= self.width || y >= self.height {
            return Err(BufferError::OutOfBounds {
                x,
                y,
                width: self.width,
                height: self.height,
            });
        }
        let idx = (y * self.width + x) * 4;
        self.data[idx] = p.r;
        self.data[idx + 1] = p.g;
        self.data[idx + 2] = p.b;
        self.data[idx + 3] = p.a;
        Ok(())
    }

    pub fn to_f32(&self) -> PixelBufferF32 {
        let mut out = Vec::with_capacity(self.data.len());
        for &byte in &self.data {
            out.push(byte as f32 / 255.0);
        }
        PixelBufferF32 {
            width: self.width,
            height: self.height,
            data: out,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PixelBufferF32 {
    pub width: usize,
    pub height: usize,
    pub data: Vec<f32>,
}

impl PixelBufferF32 {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            data: vec![0.0f32; width * height * 4],
        }
    }

    pub fn from_raw(width: usize, height: usize, data: Vec<f32>) -> Result<Self, BufferError> {
        let expected = width * height * 4;
        if data.len() != expected {
            return Err(BufferError::DimensionMismatch {
                expected,
                actual: data.len(),
            });
        }
        Ok(Self {
            width,
            height,
            data,
        })
    }

    #[inline(always)]
    pub fn dimensions(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    #[inline(always)]
    pub fn pixel_count(&self) -> usize {
        self.width * self.height
    }

    #[inline(always)]
    pub fn get_pixel(&self, x: usize, y: usize) -> Option<RgbaF32> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let idx = (y * self.width + x) * 4;
        Some(RgbaF32 {
            r: self.data[idx],
            g: self.data[idx + 1],
            b: self.data[idx + 2],
            a: self.data[idx + 3],
        })
    }

    #[inline(always)]
    pub fn set_pixel(&mut self, x: usize, y: usize, p: RgbaF32) -> Result<(), BufferError> {
        if x >= self.width || y >= self.height {
            return Err(BufferError::OutOfBounds {
                x,
                y,
                width: self.width,
                height: self.height,
            });
        }
        let idx = (y * self.width + x) * 4;
        self.data[idx] = p.r;
        self.data[idx + 1] = p.g;
        self.data[idx + 2] = p.b;
        self.data[idx + 3] = p.a;
        Ok(())
    }

    pub fn to_rgba8(&self) -> PixelBufferRgba8 {
        let mut out = Vec::with_capacity(self.data.len());
        for &val in &self.data {
            let b = (val.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            out.push(b);
        }
        PixelBufferRgba8 {
            width: self.width,
            height: self.height,
            data: out,
        }
    }
}
