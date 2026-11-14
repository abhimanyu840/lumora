use lumora_ai::cache::AiRasterCache;
use lumora_ai::models::{
    DnCnnModel, MidasModel, RealEsrganModel, SceneCategory, SceneClassifier, ZeroDceModel,
};
use lumora_ai::pipeline::{AiDenoiseOp, AiDepthBokehOp, AiSuperResOp, AiZeroDceOp};
use lumora_ai::runtime::{AiRuntimeManager, ExecutionProvider};
use lumora_ai::tensor::{Tensor, TensorError};
use lumora_core::buffer::{PixelBufferF32, RgbaF32};
use lumora_core::dag::{ExposureOp, PipelineDAG};
use std::sync::{Arc, RwLock};

fn create_test_pattern(width: usize, height: usize) -> PixelBufferF32 {
    let mut buffer = PixelBufferF32::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let r = (x as f32 / width as f32).clamp(0.0, 1.0);
            let g = (y as f32 / height as f32).clamp(0.0, 1.0);
            let b = ((x + y) as f32 / (width + height) as f32).clamp(0.0, 1.0);
            let _ = buffer.set_pixel(x, y, RgbaF32::new(r, g, b, 1.0));
        }
    }
    buffer
}

#[test]
fn test_tensor_pixel_buffer_roundtrip() {
    let buffer = create_test_pattern(16, 16);
    let tensor = Tensor::from_pixel_buffer(&buffer);

    assert_eq!(tensor.shape, vec![1, 3, 16, 16]);
    assert_eq!(tensor.len(), 3 * 16 * 16);

    let restored = tensor
        .to_pixel_buffer()
        .expect("Conversion to PixelBufferF32 should succeed");
    assert_eq!(restored.width, 16);
    assert_eq!(restored.height, 16);

    for (p1, p2) in buffer.data.iter().zip(restored.data.iter()) {
        assert!(
            (p1 - p2).abs() < 1e-4,
            "Restored pixel should match original"
        );
    }
}

#[test]
fn test_tensor_normalize_denormalize() {
    let buffer = create_test_pattern(8, 8);
    let mut tensor = Tensor::from_pixel_buffer(&buffer);
    let orig_data = tensor.data.clone();

    let mean = [0.485, 0.456, 0.406];
    let std = [0.229, 0.224, 0.225];

    tensor
        .normalize(&mean, &std)
        .expect("Normalization should succeed");
    // Ensure values changed
    assert_ne!(tensor.data[0], orig_data[0]);

    tensor
        .denormalize(&mean, &std)
        .expect("Denormalization should succeed");
    for (v_orig, v_restored) in orig_data.iter().zip(tensor.data.iter()) {
        assert!(
            (v_orig - v_restored).abs() < 1e-4,
            "Value should match after denormalization"
        );
    }
}

#[test]
fn test_tensor_bilinear_resize() {
    let buffer = create_test_pattern(8, 8);
    let tensor = Tensor::from_pixel_buffer(&buffer);

    let downsampled = tensor
        .resize_bilinear(4, 4)
        .expect("Downsampling should succeed");
    assert_eq!(downsampled.shape, vec![1, 3, 4, 4]);

    let upsampled = tensor
        .resize_bilinear(16, 16)
        .expect("Upsampling should succeed");
    assert_eq!(upsampled.shape, vec![1, 3, 16, 16]);
}

#[test]
fn test_dncnn_denoising() {
    let mut buffer = create_test_pattern(32, 32);

    // Inject pseudo-random noise into buffer
    for (i, val) in buffer.data.iter_mut().enumerate() {
        if i % 4 != 3 {
            // Keep alpha untouched
            let noise = ((i * 7 + 13) % 23) as f32 / 100.0 - 0.11;
            *val = (*val + noise).clamp(0.0, 1.0);
        }
    }

    let model = DnCnnModel::new();

    // Strength 0.0 should return identical buffer
    let untouched = model.denoise(&buffer, 0.0);
    assert_eq!(untouched.data, buffer.data);

    // Strength 1.0 should smooth out high-frequency noise
    let denoised = model.denoise(&buffer, 1.0);
    assert_eq!(denoised.width, 32);
    assert_eq!(denoised.height, 32);

    // Compute variance of difference (noise level)
    let mut diff_sum = 0.0f32;
    for (orig, clean) in buffer.data.iter().zip(denoised.data.iter()) {
        diff_sum += (orig - clean).abs();
    }
    assert!(
        diff_sum > 0.1,
        "Denoising should have modified noisy pixels"
    );
}

#[test]
fn test_midas_depth_estimation_and_bokeh() {
    let buffer = create_test_pattern(32, 32);
    let model = MidasModel::new();

    let depth = model.estimate_depth(&buffer);
    assert_eq!(depth.width, 32);
    assert_eq!(depth.height, 32);

    // Check all depth values are bounded in [0.0, 1.0]
    for &d in &depth.data {
        assert!(
            (0.0..=1.0).contains(&d),
            "Depth must be in [0.0, 1.0], got {d}"
        );
    }

    // Simulate bokeh with focal plane at 0.5 and wide aperture
    let bokeh = model.simulate_bokeh(&buffer, &depth, 0.5, 0.8, 4.0);
    assert_eq!(bokeh.width, 32);
    assert_eq!(bokeh.height, 32);

    // Pixels near focal plane should remain nearly identical, while out-of-focus should blur
    let center_pixel = buffer.get_pixel(16, 16).unwrap();
    let center_bokeh = bokeh.get_pixel(16, 16).unwrap();
    assert!((center_pixel.r - center_bokeh.r).abs() < 0.2);
}

#[test]
fn test_scene_classifier_and_preset_recommendation() {
    let classifier = SceneClassifier::new();

    // 1. Test Sunset Scene: high warmth in upper half
    let mut sunset_buf = PixelBufferF32::new(32, 32);
    for y in 0..32 {
        for x in 0..32 {
            let r = if y < 16 { 0.95 } else { 0.4 };
            let g = if y < 16 { 0.50 } else { 0.3 };
            let b = if y < 16 { 0.10 } else { 0.2 };
            let _ = sunset_buf.set_pixel(x, y, RgbaF32::new(r, g, b, 1.0));
        }
    }
    let pred_sunset = classifier.classify(&sunset_buf);
    assert_eq!(pred_sunset.category, SceneCategory::Sunset);
    assert!(pred_sunset.confidence > 0.2);

    let preset = classifier.recommend_preset(&pred_sunset);
    assert!(preset.name.contains("Golden Hour"));
    assert!(
        preset.temperature > 0.0,
        "Sunset preset should warm temperature"
    );

    // 2. Test Night Scene: low luminance across the board
    let mut night_buf = PixelBufferF32::new(32, 32);
    for y in 0..32 {
        for x in 0..32 {
            let _ = night_buf.set_pixel(x, y, RgbaF32::new(0.04, 0.05, 0.08, 1.0));
        }
    }
    let pred_night = classifier.classify(&night_buf);
    assert_eq!(pred_night.category, SceneCategory::Night);
    let night_preset = classifier.recommend_preset(&pred_night);
    assert!(
        night_preset.exposure > 0.0,
        "Night preset should boost exposure"
    );
}

#[test]
fn test_realesrgan_4x_upscale() {
    let src = create_test_pattern(16, 16);
    let model = RealEsrganModel::new();

    let upscaled = model.upscale_4x(&src);
    assert_eq!(upscaled.width, 64, "16 * 4 must equal 64");
    assert_eq!(upscaled.height, 64, "16 * 4 must equal 64");
    assert_eq!(upscaled.data.len(), 64 * 64 * 4);

    // Verify all pixels are validly clamped in [0.0, 1.0]
    for &val in &upscaled.data {
        assert!(
            (0.0..=1.0).contains(&val),
            "Upscaled pixel must be bounded: {val}"
        );
    }
}

#[test]
fn test_zerodce_low_light_enhancement() {
    // Create an underexposed dark image (luminance ~ 0.08)
    let mut dark_buf = PixelBufferF32::new(32, 32);
    for y in 0..32 {
        for x in 0..32 {
            let _ = dark_buf.set_pixel(x, y, RgbaF32::new(0.08, 0.08, 0.08, 1.0));
        }
    }

    let model = ZeroDceModel::new();
    let enhanced = model.enhance(&dark_buf, 1.0);

    assert_eq!(enhanced.width, 32);
    assert_eq!(enhanced.height, 32);

    let orig_p = dark_buf.get_pixel(16, 16).unwrap();
    let enh_p = enhanced.get_pixel(16, 16).unwrap();

    assert!(
        enh_p.r > orig_p.r,
        "Enhanced red ({}) should be brighter than original ({})",
        enh_p.r,
        orig_p.r
    );
    assert!(
        enh_p.r <= 1.0,
        "Enhanced pixel must not exceed maximum clamp 1.0"
    );
}

#[test]
fn test_ai_raster_cache_lru_and_eviction() {
    let mut cache = AiRasterCache::new::<&str>(2, None);

    let buf1 = create_test_pattern(8, 8);
    let buf2 = create_test_pattern(16, 16);
    let buf3 = create_test_pattern(32, 32);

    cache.insert("key1".to_string(), buf1);
    cache.insert("key2".to_string(), buf2);
    assert_eq!(cache.len(), 2);
    assert!(cache.memory_usage_bytes() > 0);

    // Access key1 to make it most-recently used
    assert!(cache.get("key1").is_some());

    // Insert key3 -> should evict key2 (the LRU entry)
    cache.insert("key3".to_string(), buf3);
    assert_eq!(cache.len(), 2);

    assert!(cache.get("key1").is_some(), "key1 should remain in cache");
    assert!(cache.get("key3").is_some(), "key3 should remain in cache");
    assert!(cache.get("key2").is_none(), "key2 should have been evicted");

    // Invalidation
    cache.invalidate_prefix("key");
    assert_eq!(cache.len(), 0);
}

#[test]
fn test_pipeline_dag_cached_ai_nodes() {
    let source = create_test_pattern(16, 16);
    let cache = Arc::new(RwLock::new(AiRasterCache::new::<&str>(4, None)));

    let mut dag = PipelineDAG::new();

    // Node 1: AI Zero-DCE low-light enhancement with cache
    let ai_node_id = dag.add_node(AiZeroDceOp::with_cache(1.0, Arc::clone(&cache)));

    // Node 2: Exposure adjustment (+0.5 EV)
    let exp_node_id = dag.add_node(ExposureOp::new(0.5));

    // First render pass
    let render1 = dag.render(&source);
    assert_eq!(render1.width, 16);
    assert_eq!(render1.height, 16);

    // Verify AI pass populated the cache
    {
        let c = cache.read().unwrap();
        assert_eq!(c.len(), 1, "Cache should contain the AI output");
    }

    // Now update downstream Exposure slider (+1.0 EV)
    dag.update_node(exp_node_id, ExposureOp::new(1.0));

    // Render using cache: should leverage cached AI output
    let render2 = dag.render_cached(&source);
    assert_eq!(render2.width, 16);
    assert_eq!(render2.height, 16);

    // Verify AI node is still in DAG
    assert!(dag.get_node(ai_node_id).is_some());
    assert!(dag.get_node(exp_node_id).is_some());
}

#[test]
fn test_pipeline_dag_all_ai_ops() {
    let source = create_test_pattern(16, 16);
    let mut dag = PipelineDAG::new();

    dag.add_node(AiDenoiseOp::new(0.5));
    dag.add_node(AiDepthBokehOp::new(0.5, 0.4, 2.0));
    dag.add_node(AiZeroDceOp::new(0.8));

    let output = dag.render(&source);
    assert_eq!(output.width, 16);
    assert_eq!(output.height, 16);

    // Test super-res in DAG
    let mut sr_dag = PipelineDAG::new();
    sr_dag.add_node(AiSuperResOp::new());
    let upscaled = sr_dag.render(&source);
    assert_eq!(upscaled.width, 64);
    assert_eq!(upscaled.height, 64);
}

#[test]
fn test_ai_session_and_runtime_manager() {
    let temp_dir = std::env::temp_dir().join("lumora_models_test");
    let _ = std::fs::create_dir_all(&temp_dir);
    let manager = AiRuntimeManager::new(&temp_dir, ExecutionProvider::Cpu);

    let models = [
        "dncnn",
        "midas",
        "mobilenet_scene",
        "realesrgan_x4",
        "zerodce",
    ];
    let input_buf = create_test_pattern(16, 16);
    let input_tensor = Tensor::from_pixel_buffer(&input_buf);

    for &name in &models {
        let session = manager
            .create_session(name)
            .unwrap_or_else(|e| panic!("Session creation failed for {name}: {e}"));

        assert_eq!(session.metadata().provider, ExecutionProvider::Cpu);
        assert!(session.metadata().quantized);

        let output_tensor = session
            .run(&input_tensor)
            .unwrap_or_else(|e| panic!("Session run failed for {name}: {e}"));

        assert!(
            !output_tensor.is_empty(),
            "Output tensor for {name} should not be empty"
        );

        match name {
            "dncnn" | "zerodce" => {
                assert_eq!(output_tensor.shape, vec![1, 3, 16, 16]);
            }
            "midas" => {
                assert_eq!(output_tensor.shape, vec![1, 1, 16, 16]);
            }
            "mobilenet_scene" => {
                assert_eq!(output_tensor.shape, vec![1, 6]);
                let sum: f32 = output_tensor.data.iter().sum();
                assert!((sum - 1.0).abs() < 1e-3, "Scene scores must sum to 1.0");
            }
            "realesrgan_x4" => {
                assert_eq!(output_tensor.shape, vec![1, 3, 64, 64]);
            }
            _ => unreachable!(),
        }
    }

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_ai_raster_cache_disk_persistence() {
    let temp_cache_dir = std::env::temp_dir().join("lumora_cache_test_disk");
    let _ = std::fs::remove_dir_all(&temp_cache_dir);
    let _ = std::fs::create_dir_all(&temp_cache_dir);

    // Cache with capacity of 1 in RAM, with disk directory
    let mut cache = AiRasterCache::new(1, Some(&temp_cache_dir));

    let buf1 = create_test_pattern(8, 8);
    let buf2 = create_test_pattern(12, 12);

    cache.insert("node_0.50".to_string(), buf1.clone());
    assert_eq!(cache.len(), 1);

    // Inserting second buffer evicts buf1 to disk
    cache.insert("node_1.00".to_string(), buf2);
    assert_eq!(cache.len(), 1);

    // buf1 should be reloaded from disk cache in binary format
    let reloaded = cache.get("node_0.50");
    assert!(
        reloaded.is_some(),
        "Evicted buffer should be reloaded from disk"
    );
    let reloaded_buf = reloaded.unwrap();
    assert_eq!(reloaded_buf.width, 8);
    assert_eq!(reloaded_buf.height, 8);
    for (p1, p2) in buf1.data.iter().zip(reloaded_buf.data.iter()) {
        assert_eq!(
            p1, p2,
            "Binary restored pixel values must match bit-for-bit"
        );
    }

    // Invalidate prefix with special characters (dot)
    cache.invalidate_prefix("node_0.");
    assert!(
        cache.get("node_0.50").is_none(),
        "Invalidated entry must be removed from RAM and disk"
    );

    cache.clear();
    assert_eq!(cache.len(), 0);
    let _ = std::fs::remove_dir_all(&temp_cache_dir);
}

#[test]
fn test_midas_bokeh_mask_generation() {
    let buffer = create_test_pattern(16, 16);
    let model = MidasModel::new();
    let depth = model.estimate_depth(&buffer);

    let bokeh_mask = model.compute_bokeh_mask(&depth, 0.6, 0.8);
    assert_eq!(bokeh_mask.width, 16);
    assert_eq!(bokeh_mask.height, 16);

    for &val in &bokeh_mask.data {
        assert!(
            (0.0..=1.0).contains(&val),
            "Bokeh mask value must be in [0, 1], got {val}"
        );
    }
}

#[test]
fn test_super_res_alpha_preservation() {
    let mut buffer = PixelBufferF32::new(8, 8);
    for y in 0..8 {
        for x in 0..8 {
            let alpha = if x < 4 { 0.25 } else { 0.75 };
            let _ = buffer.set_pixel(x, y, RgbaF32::new(0.5, 0.5, 0.5, alpha));
        }
    }

    let model = RealEsrganModel::new();
    let upscaled = model.upscale_4x(&buffer);
    assert_eq!(upscaled.width, 32);
    assert_eq!(upscaled.height, 32);

    let left_alpha = upscaled.get_pixel(4, 4).unwrap().a;
    let right_alpha = upscaled.get_pixel(28, 4).unwrap().a;

    assert!(
        (left_alpha - 0.25).abs() < 0.1,
        "Left side alpha should remain close to 0.25, got {left_alpha}"
    );
    assert!(
        (right_alpha - 0.75).abs() < 0.1,
        "Right side alpha should remain close to 0.75, got {right_alpha}"
    );
}

#[test]
fn test_tensor_2d_depth_conversion() {
    let depth_data = vec![0.5f32; 16];
    let tensor_2d = Tensor::from_data(vec![4, 4], depth_data).unwrap();
    let buffer = tensor_2d
        .to_pixel_buffer()
        .expect("2D tensor should convert to PixelBufferF32");
    assert_eq!(buffer.width, 4);
    assert_eq!(buffer.height, 4);
    assert_eq!(buffer.data.len(), 4 * 4 * 4);
    assert_eq!(buffer.data[0], 0.5);
    assert_eq!(buffer.data[3], 1.0); // Alpha channel
}

#[test]
fn test_tensor_resize_bilinear_bounds_and_empty() {
    let buffer = create_test_pattern(1, 1);
    let tensor = Tensor::from_pixel_buffer(&buffer);

    // Single pixel to 4x4
    let upscaled = tensor.resize_bilinear(4, 4).unwrap();
    assert_eq!(upscaled.shape, vec![1, 3, 4, 4]);

    // Non-square aspect ratio: 16x1 to 1x16
    let buf_strip = create_test_pattern(16, 1);
    let tensor_strip = Tensor::from_pixel_buffer(&buf_strip);
    let rotated = tensor_strip.resize_bilinear(16, 1).unwrap();
    assert_eq!(rotated.shape, vec![1, 3, 16, 1]);

    // Zero dimensions should error rather than panic
    let empty_tensor = Tensor::zeros(vec![1, 3, 0, 0]);
    let err = empty_tensor.resize_bilinear(4, 4);
    assert!(matches!(err, Err(TensorError::InvalidDimensions(_))));

    let err_target = tensor.resize_bilinear(0, 4);
    assert!(matches!(err_target, Err(TensorError::InvalidDimensions(_))));
}

#[test]
fn test_empty_buffers_across_all_models() {
    let empty = PixelBufferF32::new(0, 0);

    let dncnn = DnCnnModel::new();
    let denoised = dncnn.denoise(&empty, 1.0);
    assert_eq!(denoised.width, 0);

    let midas = MidasModel::new();
    let depth = midas.estimate_depth(&empty);
    assert_eq!(depth.width, 0);
    let bokeh_mask = midas.compute_bokeh_mask(&depth, 0.5, 0.5);
    assert_eq!(bokeh_mask.width, 0);

    let classifier = SceneClassifier::new();
    let pred = classifier.classify(&empty);
    assert_eq!(pred.scores.len(), 6);

    let sr = RealEsrganModel::new();
    let upscaled = sr.upscale_4x(&empty);
    assert_eq!(upscaled.width, 0);

    let zerodce = ZeroDceModel::new();
    let enhanced = zerodce.enhance(&empty, 1.0);
    assert_eq!(enhanced.width, 0);
}
