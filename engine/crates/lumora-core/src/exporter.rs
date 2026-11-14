//! Memory-efficient tiled high-resolution image exporter for Lumora engine.
//!
//! Provides the `ImageExporter` pipeline supporting tile-based processing (e.g. 512x512
//! or 1024x1024 tiles) with Rayon parallelization. Allows exporting 12MP–48MP mobile photos
//! to JPEG, PNG, and WebP formats with strictly bounded memory usage.

use std::fmt;
use std::fs::File;
use std::io::{Cursor, Write};
use std::path::Path;

use image::codecs::jpeg::JpegEncoder;
use image::codecs::png::PngEncoder;
use image::codecs::webp::WebPEncoder;
use image::{ExtendedColorType, ImageEncoder};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::buffer::{PixelBufferF32, PixelBufferRgba8};
use crate::dag::PipelineDAG;

/// Supported image export formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExportFormat {
    /// JPEG compression with quality factor (1..=100, default 90).
    Jpeg { quality: u8 },
    /// Lossless PNG encoding.
    Png,
    /// Lossless WebP encoding.
    WebP { lossless: bool },
}

impl Default for ExportFormat {
    fn default() -> Self {
        Self::Jpeg { quality: 90 }
    }
}

/// Padding margins around an extracted sub-image tile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct TilePadding {
    pub left: usize,
    pub right: usize,
    pub top: usize,
    pub bottom: usize,
}

impl TilePadding {
    pub fn new(left: usize, right: usize, top: usize, bottom: usize) -> Self {
        Self {
            left,
            right,
            top,
            bottom,
        }
    }
}

/// Errors occurring during image export or encoding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExportError {
    EmptyBuffer,
    InvalidDimensions { width: usize, height: usize },
    InvalidTileSize(usize),
    InvalidQuality(u8),
    IoError(String),
    EncodingError(String),
}

impl fmt::Display for ExportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyBuffer => write!(f, "Cannot export empty or zero-dimension image buffer"),
            Self::InvalidDimensions { width, height } => {
                write!(f, "Invalid image dimensions for export: {width}x{height}")
            }
            Self::InvalidTileSize(sz) => write!(f, "Tile size must be greater than zero, got {sz}"),
            Self::InvalidQuality(q) => {
                write!(
                    f,
                    "JPEG quality must be between 1 and 100 inclusive, got {q}"
                )
            }
            Self::IoError(msg) => write!(f, "I/O error during export: {msg}"),
            Self::EncodingError(msg) => write!(f, "Encoding error during export: {msg}"),
        }
    }
}

impl std::error::Error for ExportError {}

/// Memory-efficient tiled high-resolution image exporter.
#[derive(Debug, Clone)]
pub struct ImageExporter {
    format: ExportFormat,
    tile_size: usize,
}

impl Default for ImageExporter {
    fn default() -> Self {
        Self::new(ExportFormat::default())
    }
}

impl ImageExporter {
    /// Create a new exporter for the specified target format with default 512x512 tile size.
    pub fn new(format: ExportFormat) -> Self {
        Self {
            format,
            tile_size: 512,
        }
    }

    /// Configure the tile dimension (e.g. 256, 512, 1024 pixels).
    pub fn with_tile_size(mut self, tile_size: usize) -> Self {
        self.tile_size = if tile_size > 0 { tile_size } else { 512 };
        self
    }

    /// Retrieve the configured tile size.
    pub fn tile_size(&self) -> usize {
        self.tile_size
    }

    /// Retrieve the target export format.
    pub fn format(&self) -> ExportFormat {
        self.format
    }

    /// Extracts a sub-region tile from a full-resolution float buffer with bounds protection.
    pub fn extract_tile(
        source: &PixelBufferF32,
        x_offset: usize,
        y_offset: usize,
        tile_w: usize,
        tile_h: usize,
    ) -> PixelBufferF32 {
        let mut tile = PixelBufferF32::new(tile_w, tile_h);
        let src_width = source.width;
        let src_height = source.height;

        for r in 0..tile_h {
            let cur_y = y_offset + r;
            if cur_y >= src_height {
                break;
            }
            if x_offset >= src_width {
                continue;
            }
            let copy_w = tile_w.min(src_width - x_offset);
            let src_start = (cur_y * src_width + x_offset) * 4;
            let src_end = src_start + copy_w * 4;
            let dst_start = r * tile_w * 4;
            tile.data[dst_start..dst_start + copy_w * 4]
                .copy_from_slice(&source.data[src_start..src_end]);
        }

        tile
    }

    /// Extracts a sub-region tile from a full-resolution float buffer with an apron (halo) of pixels.
    pub fn extract_tile_with_apron(
        source: &PixelBufferF32,
        x_offset: usize,
        y_offset: usize,
        tile_w: usize,
        tile_h: usize,
        padding: TilePadding,
    ) -> PixelBufferF32 {
        let padded_w = padding.left + tile_w + padding.right;
        let padded_h = padding.top + tile_h + padding.bottom;
        let mut tile = PixelBufferF32::new(padded_w, padded_h);

        let src_x0 = x_offset.saturating_sub(padding.left);
        let src_y0 = y_offset.saturating_sub(padding.top);
        let src_w = source.width;
        let src_h = source.height;

        for r in 0..padded_h {
            let cur_y = src_y0 + r;
            if cur_y >= src_h {
                break;
            }
            if src_x0 >= src_w {
                continue;
            }
            let copy_w = padded_w.min(src_w - src_x0);
            let src_start = (cur_y * src_w + src_x0) * 4;
            let src_end = src_start + copy_w * 4;
            let dst_start = r * padded_w * 4;
            tile.data[dst_start..dst_start + copy_w * 4]
                .copy_from_slice(&source.data[src_start..src_end]);
        }

        tile
    }

    /// Renders a full-resolution float buffer into an RGBA8 buffer using parallel tiled processing.
    ///
    /// Memory consumption is strictly bounded to `num_threads * tile_size * tile_size * 16` bytes.
    pub fn render_tiled_f32_to_rgba8(
        &self,
        buffer: &PixelBufferF32,
    ) -> Result<PixelBufferRgba8, ExportError> {
        let width = buffer.width;
        let height = buffer.height;
        if width == 0 || height == 0 || buffer.data.is_empty() {
            return Err(ExportError::EmptyBuffer);
        }

        let mut output = PixelBufferRgba8::new(width, height);
        let tile_size = self.tile_size;
        let tiles_x = width.div_ceil(tile_size);
        let row_stride = width * 4;
        let band_chunk_size = tile_size * row_stride;

        output
            .data
            .par_chunks_mut(band_chunk_size)
            .enumerate()
            .for_each(|(ty, band_slice)| {
                let y_offset = ty * tile_size;
                let band_h = band_slice.len() / row_stride;

                for tx in 0..tiles_x {
                    let x_offset = tx * tile_size;
                    let tile_w = (width - x_offset).min(tile_size);

                    let tile = Self::extract_tile(buffer, x_offset, y_offset, tile_w, band_h);

                    for r in 0..band_h {
                        let dst_start = r * row_stride + x_offset * 4;
                        let src_start = r * tile_w * 4;
                        for x in 0..tile_w {
                            let s_idx = src_start + x * 4;
                            let d_idx = dst_start + x * 4;
                            band_slice[d_idx] =
                                (tile.data[s_idx].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                            band_slice[d_idx + 1] =
                                (tile.data[s_idx + 1].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                            band_slice[d_idx + 2] =
                                (tile.data[s_idx + 2].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                            band_slice[d_idx + 3] =
                                (tile.data[s_idx + 3].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                        }
                    }
                }
            });

        Ok(output)
    }

    /// Evaluates an entire non-destructive DAG pipeline tile-by-tile over a full-resolution buffer.
    ///
    /// Memory consumption is strictly bounded during execution, eliminating OOM spikes on mobile.
    pub fn render_tiled_dag(
        &self,
        source: &PixelBufferF32,
        dag: &PipelineDAG,
    ) -> Result<PixelBufferRgba8, ExportError> {
        let width = source.width;
        let height = source.height;
        if width == 0 || height == 0 || source.data.is_empty() {
            return Err(ExportError::EmptyBuffer);
        }

        let mut output = PixelBufferRgba8::new(width, height);

        // Pre-resolve pipeline dependencies (e.g. U-Point target colors sampled after upstream DAG ops)
        let mut upstream_ops: Vec<&dyn crate::dag::ImageOp> = Vec::new();
        for node in dag.nodes() {
            if node.enabled {
                node.op.prepare_pipeline(source, &upstream_ops);
                upstream_ops.push(&*node.op);
            }
        }

        let tile_size = self.tile_size;
        let tiles_x = width.div_ceil(tile_size);
        let row_stride = width * 4;
        let band_chunk_size = tile_size * row_stride;

        output
            .data
            .par_chunks_mut(band_chunk_size)
            .enumerate()
            .for_each(|(ty, band_slice)| {
                let y_offset = ty * tile_size;
                let band_h = band_slice.len() / row_stride;

                for tx in 0..tiles_x {
                    let x_offset = tx * tile_size;
                    let tile_w = (width - x_offset).min(tile_size);

                    let padding = TilePadding {
                        left: x_offset.min(1),
                        right: (width - (x_offset + tile_w)).min(1),
                        top: y_offset.min(1),
                        bottom: (height - (y_offset + band_h)).min(1),
                    };

                    let mut tile = Self::extract_tile_with_apron(
                        source, x_offset, y_offset, tile_w, band_h, padding,
                    );

                    let src_origin_x = x_offset.saturating_sub(padding.left);
                    let src_origin_y = y_offset.saturating_sub(padding.top);

                    // Apply every enabled operation in the DAG to this apron-padded tile
                    for node in dag.nodes() {
                        if node.enabled {
                            node.op.apply_tile_with_source(
                                &mut tile,
                                src_origin_x,
                                src_origin_y,
                                width,
                                height,
                                Some(source),
                            );
                        }
                    }

                    // Convert tile float pixels to RGBA8 and copy directly into destination slice
                    let padded_w = tile.width;
                    for r in 0..band_h {
                        let dst_start = r * row_stride + x_offset * 4;
                        let src_start = ((padding.top + r) * padded_w + padding.left) * 4;
                        for x in 0..tile_w {
                            let s_idx = src_start + x * 4;
                            let d_idx = dst_start + x * 4;
                            band_slice[d_idx] =
                                (tile.data[s_idx].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                            band_slice[d_idx + 1] =
                                (tile.data[s_idx + 1].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                            band_slice[d_idx + 2] =
                                (tile.data[s_idx + 2].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                            band_slice[d_idx + 3] =
                                (tile.data[s_idx + 3].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                        }
                    }
                }
            });

        Ok(output)
    }

    /// Encode an RGBA8 buffer to encoded bytes in the target format (JPEG, PNG, WebP).
    pub fn encode_rgba8(
        &self,
        width: usize,
        height: usize,
        rgba_data: &[u8],
    ) -> Result<Vec<u8>, ExportError> {
        if width == 0 || height == 0 || rgba_data.is_empty() {
            return Err(ExportError::EmptyBuffer);
        }
        if rgba_data.len() != width * height * 4 {
            return Err(ExportError::InvalidDimensions { width, height });
        }

        let mut output = Cursor::new(Vec::new());

        match self.format {
            ExportFormat::Jpeg { quality } => {
                if quality == 0 || quality > 100 {
                    return Err(ExportError::InvalidQuality(quality));
                }
                // JPEG does not support alpha; composite transparent areas against clean white matte
                let mut rgb_data = Vec::with_capacity(width * height * 3);
                for chunk in rgba_data.chunks_exact(4) {
                    if chunk[3] == 255 {
                        rgb_data.push(chunk[0]);
                        rgb_data.push(chunk[1]);
                        rgb_data.push(chunk[2]);
                    } else {
                        let alpha = chunk[3] as f32 / 255.0;
                        let inv_alpha = 1.0 - alpha;
                        let r = (chunk[0] as f32 * alpha + 255.0 * inv_alpha + 0.5) as u8;
                        let g = (chunk[1] as f32 * alpha + 255.0 * inv_alpha + 0.5) as u8;
                        let b = (chunk[2] as f32 * alpha + 255.0 * inv_alpha + 0.5) as u8;
                        rgb_data.push(r);
                        rgb_data.push(g);
                        rgb_data.push(b);
                    }
                }
                let mut encoder = JpegEncoder::new_with_quality(&mut output, quality);
                encoder
                    .encode(
                        &rgb_data,
                        width as u32,
                        height as u32,
                        ExtendedColorType::Rgb8,
                    )
                    .map_err(|e| ExportError::EncodingError(e.to_string()))?;
            }
            ExportFormat::Png => {
                let encoder = PngEncoder::new(&mut output);
                encoder
                    .write_image(
                        rgba_data,
                        width as u32,
                        height as u32,
                        ExtendedColorType::Rgba8,
                    )
                    .map_err(|e| ExportError::EncodingError(e.to_string()))?;
            }
            ExportFormat::WebP { lossless: _ } => {
                let encoder = WebPEncoder::new_lossless(&mut output);
                encoder
                    .write_image(
                        rgba_data,
                        width as u32,
                        height as u32,
                        ExtendedColorType::Rgba8,
                    )
                    .map_err(|e| ExportError::EncodingError(e.to_string()))?;
            }
        }

        Ok(output.into_inner())
    }

    /// Export an RGBA8 buffer directly to encoded bytes.
    pub fn export_rgba8(&self, buffer: &PixelBufferRgba8) -> Result<Vec<u8>, ExportError> {
        self.encode_rgba8(buffer.width, buffer.height, &buffer.data)
    }

    /// Export a float buffer using memory-efficient tiled processing to encoded bytes.
    pub fn export_f32(&self, buffer: &PixelBufferF32) -> Result<Vec<u8>, ExportError> {
        let rgba = self.render_tiled_f32_to_rgba8(buffer)?;
        self.export_rgba8(&rgba)
    }

    /// Export a full-resolution float buffer processed through a PipelineDAG to encoded bytes.
    pub fn export_tiled_dag(
        &self,
        source: &PixelBufferF32,
        dag: &PipelineDAG,
    ) -> Result<Vec<u8>, ExportError> {
        let rgba = self.render_tiled_dag(source, dag)?;
        self.export_rgba8(&rgba)
    }

    /// Export an image directly to a file on disk.
    pub fn export_to_file(
        &self,
        buffer: &PixelBufferF32,
        path: impl AsRef<Path>,
    ) -> Result<usize, ExportError> {
        let bytes = self.export_f32(buffer)?;
        let mut file = File::create(path.as_ref())
            .map_err(|e| ExportError::IoError(format!("Failed to create file: {e}")))?;
        file.write_all(&bytes)
            .map_err(|e| ExportError::IoError(format!("Failed to write file: {e}")))?;
        Ok(bytes.len())
    }

    /// Export an image DAG pipeline directly to a file on disk.
    pub fn export_dag_to_file(
        &self,
        source: &PixelBufferF32,
        dag: &PipelineDAG,
        path: impl AsRef<Path>,
    ) -> Result<usize, ExportError> {
        let bytes = self.export_tiled_dag(source, dag)?;
        let mut file = File::create(path.as_ref())
            .map_err(|e| ExportError::IoError(format!("Failed to create file: {e}")))?;
        file.write_all(&bytes)
            .map_err(|e| ExportError::IoError(format!("Failed to write file: {e}")))?;
        Ok(bytes.len())
    }
}
