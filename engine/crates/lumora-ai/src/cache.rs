use lumora_core::buffer::PixelBufferF32;
use std::collections::VecDeque;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
struct CacheEntry {
    key: String,
    buffer: PixelBufferF32,
    size_bytes: usize,
}

const RASTER_CACHE_MAGIC: &[u8; 4] = b"LRB1"; // Lumora Raster Buffer v1

fn serialize_raster_buffer(buffer: &PixelBufferF32) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(20 + buffer.data.len() * 4);
    bytes.extend_from_slice(RASTER_CACHE_MAGIC);
    bytes.extend_from_slice(&(buffer.width as u64).to_le_bytes());
    bytes.extend_from_slice(&(buffer.height as u64).to_le_bytes());
    for &val in &buffer.data {
        bytes.extend_from_slice(&val.to_le_bytes());
    }
    bytes
}

fn deserialize_raster_buffer(bytes: &[u8]) -> Option<PixelBufferF32> {
    if bytes.len() >= 20 && &bytes[0..4] == RASTER_CACHE_MAGIC {
        let width = u64::from_le_bytes(bytes[4..12].try_into().ok()?) as usize;
        let height = u64::from_le_bytes(bytes[12..20].try_into().ok()?) as usize;
        let expected_floats = width.checked_mul(height)?.checked_mul(4)?;
        if bytes.len() != 20 + expected_floats * 4 {
            return None;
        }
        let mut data = Vec::with_capacity(expected_floats);
        for chunk in bytes[20..].chunks_exact(4) {
            data.push(f32::from_le_bytes(chunk.try_into().ok()?));
        }
        return Some(PixelBufferF32 {
            width,
            height,
            data,
        });
    }

    // JSON fallback for backwards compatibility
    serde_json::from_slice::<PixelBufferF32>(bytes).ok()
}

fn sanitize_key(key: &str) -> String {
    key.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// LRU Raster Cache for intermediate AI model outputs.
///
/// Ensures non-destructive slider operations (e.g. Lightroom exposure tweaks or
/// Snapseed U-Point adjustments) can scrub smoothly at 60 FPS on top of heavy AI passes
/// (Denoise, Bokeh, Low-Light Zero-DCE) without triggering expensive neural re-inference.
pub struct AiRasterCache {
    memory_limit_frames: usize,
    entries: VecDeque<CacheEntry>,
    disk_dir: Option<PathBuf>,
}

impl AiRasterCache {
    /// Creates a new LRU cache with the specified memory frame limit and optional disk directory.
    pub fn new<P: AsRef<Path>>(memory_limit_frames: usize, disk_dir: Option<P>) -> Self {
        let disk_path = disk_dir.map(|p| {
            let path = p.as_ref().to_path_buf();
            let _ = fs::create_dir_all(&path);
            path
        });

        Self {
            memory_limit_frames: memory_limit_frames.max(1),
            entries: VecDeque::new(),
            disk_dir: disk_path,
        }
    }

    /// Number of frames currently held in memory.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Estimated memory usage in bytes of all cached frames in RAM.
    pub fn memory_usage_bytes(&self) -> usize {
        self.entries.iter().map(|e| e.size_bytes).sum()
    }

    /// Retrieves a cached raster buffer by key, refreshing its LRU position.
    pub fn get(&mut self, key: &str) -> Option<PixelBufferF32> {
        // 1. Search in-memory entries
        if let Some(pos) = self.entries.iter().position(|e| e.key == key) {
            let entry = self.entries.remove(pos)?;
            let buffer = entry.buffer.clone();
            // Move to front (most recently used)
            self.entries.push_front(entry);
            return Some(buffer);
        }

        // 2. Fallback to disk cache if available
        if let Some(disk_path) = self.disk_key_path(key) {
            if disk_path.exists() {
                if let Ok(bytes) = fs::read(&disk_path) {
                    if let Some(buffer) = deserialize_raster_buffer(&bytes) {
                        self.insert(key.to_string(), buffer.clone());
                        return Some(buffer);
                    }
                }
            }
        }

        None
    }

    /// Inserts a new buffer into the cache, evicting the least recently used entry
    /// if capacity is exceeded.
    pub fn insert(&mut self, key: String, buffer: PixelBufferF32) {
        // If key already exists, remove it first
        if let Some(pos) = self.entries.iter().position(|e| e.key == key) {
            self.entries.remove(pos);
        }

        let size_bytes = buffer.data.len() * std::mem::size_of::<f32>();

        // Evict LRU items if at capacity
        while self.entries.len() >= self.memory_limit_frames {
            if let Some(evicted) = self.entries.pop_back() {
                self.persist_to_disk(&evicted.key, &evicted.buffer);
            }
        }

        self.entries.push_front(CacheEntry {
            key,
            buffer,
            size_bytes,
        });
    }

    /// Invalidates all entries matching the given key prefix.
    pub fn invalidate_prefix(&mut self, prefix: &str) {
        self.entries.retain(|e| !e.key.starts_with(prefix));

        if let Some(dir) = &self.disk_dir {
            let safe_prefix = sanitize_key(prefix);
            if let Ok(entries) = fs::read_dir(dir) {
                for entry in entries.flatten() {
                    let name = entry.file_name().to_string_lossy().to_string();
                    if name.starts_with(prefix) || name.starts_with(&safe_prefix) {
                        let _ = fs::remove_file(entry.path());
                    }
                }
            }
        }
    }

    /// Clears all entries from RAM and disk cache.
    pub fn clear(&mut self) {
        self.entries.clear();
        if let Some(dir) = &self.disk_dir {
            if let Ok(entries) = fs::read_dir(dir) {
                for entry in entries.flatten() {
                    let _ = fs::remove_file(entry.path());
                }
            }
        }
    }

    fn disk_key_path(&self, key: &str) -> Option<PathBuf> {
        let dir = self.disk_dir.as_ref()?;
        let safe_name = sanitize_key(key);
        Some(dir.join(format!("{safe_name}.cache")))
    }

    fn persist_to_disk(&self, key: &str, buffer: &PixelBufferF32) {
        if let Some(path) = self.disk_key_path(key) {
            let serialized = serialize_raster_buffer(buffer);
            let _ = fs::write(path, serialized);
        }
    }
}
