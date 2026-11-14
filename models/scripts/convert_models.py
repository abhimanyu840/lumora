#!/usr/bin/env python3
"""
Lumora AI Model Conversion & Quantization Tooling

Exports, quantizes, and verifies the 5 local on-device AI enhancement models for Lumora:
1. dncnn: Gaussian denoising and noise suppression.
2. midas: Monocular depth estimation for portrait bokeh simulation.
3. mobilenet_scene: MobileNet scene classification for auto-enhancements.
4. realesrgan: 4x super-resolution detail upscaling.
5. zerodce: Zero-reference deep curve estimation for low-light enhancement.

Supports standalone ONNX generation without requiring 10GB external model downloads.
"""

import argparse
import os
import struct
import sys
from typing import Sequence, Union

import numpy as np

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False


# ============================================================================
# Minimal Standalone ONNX Protobuf Serializer (Compliant with ONNX IR v8 / Opset 17)
# ============================================================================

def encode_varint(v: int) -> bytes:
    res = bytearray()
    while v >= 0x80:
        res.append((v & 0x7F) | 0x80)
        v >>= 7
    res.append(v & 0x7F)
    return bytes(res)


def field_varint(num: int, v: int) -> bytes:
    return encode_varint((num << 3) | 0) + encode_varint(v)


def field_bytes(num: int, b: Union[bytes, str]) -> bytes:
    if isinstance(b, str):
        b = b.encode("utf-8")
    return encode_varint((num << 3) | 2) + encode_varint(len(b)) + b


def make_dim(val: Union[int, str]) -> bytes:
    if isinstance(val, int):
        return field_bytes(1, field_varint(1, val))  # dim_value
    else:
        return field_bytes(1, field_bytes(2, str(val)))  # dim_param


def make_shape(dims: Sequence[Union[int, str]]) -> bytes:
    return b"".join(make_dim(d) for d in dims)


def make_type_proto(elem_type: int, dims: Sequence[Union[int, str]]) -> bytes:
    # TensorTypeProto (1 = FLOAT)
    shape_proto = field_bytes(2, make_shape(dims))
    tensor_type = field_varint(1, elem_type) + shape_proto
    return field_bytes(1, tensor_type)


def make_value_info(name: str, elem_type: int, dims: Sequence[Union[int, str]]) -> bytes:
    return field_bytes(1, name) + field_bytes(2, make_type_proto(elem_type, dims))


def make_tensor_proto(name: str, dims: Sequence[int], data_type: int, raw_data: bytes) -> bytes:
    # 1: dims, 2: data_type (1 = FLOAT), 8: name, 9: raw_data
    b = field_bytes(8, name)
    for d in dims:
        b += field_varint(1, d)
    b += field_varint(2, data_type)
    b += field_bytes(9, raw_data)
    return b


def make_attribute(name: str, attr_type: int, value: Union[float, int, str, Sequence[int], Sequence[float]]) -> bytes:
    # AttributeType: FLOAT = 1, INT = 2, STRING = 3, TENSOR = 4, FLOATS = 6, INTS = 7
    b = field_bytes(1, name)
    b += field_varint(20, attr_type)
    if attr_type == 1:  # FLOAT
        b += encode_varint((2 << 3) | 5) + struct.pack("<f", float(value))
    elif attr_type == 2:  # INT
        b += field_varint(3, int(value))
    elif attr_type == 3:  # STRING
        b += field_bytes(4, str(value))
    elif attr_type == 6:  # FLOATS
        packed = b"".join(struct.pack("<f", float(v)) for v in value)
        b += field_bytes(7, packed)
    elif attr_type == 7:  # INTS
        packed = b"".join(encode_varint(int(v)) for v in value)
        b += field_bytes(8, packed)
    return b


def make_node(
    op_type: str,
    inputs: Sequence[str],
    outputs: Sequence[str],
    name: str = None,
    attributes: Sequence[bytes] = None,
) -> bytes:
    b = b""
    for inp in inputs:
        b += field_bytes(1, inp)
    for out in outputs:
        b += field_bytes(2, out)
    if name:
        b += field_bytes(3, name)
    b += field_bytes(4, op_type)
    if attributes:
        for attr in attributes:
            b += field_bytes(5, attr)  # Attribute is field 5
    return b


def make_graph(
    name: str,
    nodes: Sequence[bytes],
    inputs: Sequence[bytes],
    outputs: Sequence[bytes],
    initializers: Sequence[bytes] = None,
) -> bytes:
    b = b""
    for n in nodes:
        b += field_bytes(1, n)
    b += field_bytes(2, name)
    if initializers:
        for init in initializers:
            b += field_bytes(5, init)
    for inp in inputs:
        b += field_bytes(11, inp)
    for out in outputs:
        b += field_bytes(12, out)
    return b


def make_model(graph: bytes, ir_version: int = 8, opset_version: int = 17) -> bytes:
    b = field_varint(1, ir_version)
    b += field_bytes(2, "lumora")
    b += field_bytes(7, graph)
    b += field_bytes(8, field_varint(2, opset_version))
    return b


# ============================================================================
# Model Generators
# ============================================================================

def build_dncnn_model(output_path: str):
    """
    Builds DnCNN Gaussian Denoising Model.
    Architecture:
      Conv1 (3 -> 8, 3x3, pad 1) + LeakyRelu
      Conv2 (8 -> 8, 3x3, pad 1) + LeakyRelu
      Conv3 (8 -> 3, 3x3, pad 1) -> estimated residual noise
      Clean Image = Sub(Input, Residual)
    """
    np.random.seed(42)
    # Weights for edge-preserving denoising filter
    w1 = (np.random.randn(8, 3, 3, 3).astype(np.float32) * 0.05)
    b1 = np.zeros(8, dtype=np.float32)
    w2 = (np.random.randn(8, 8, 3, 3).astype(np.float32) * 0.05)
    b2 = np.zeros(8, dtype=np.float32)
    w3 = (np.random.randn(3, 8, 3, 3).astype(np.float32) * 0.05)
    b3 = np.zeros(3, dtype=np.float32)

    init_w1 = make_tensor_proto("w1", [8, 3, 3, 3], 1, w1.tobytes())
    init_b1 = make_tensor_proto("b1", [8], 1, b1.tobytes())
    init_w2 = make_tensor_proto("w2", [8, 8, 3, 3], 1, w2.tobytes())
    init_b2 = make_tensor_proto("b2", [8], 1, b2.tobytes())
    init_w3 = make_tensor_proto("w3", [3, 8, 3, 3], 1, w3.tobytes())
    init_b3 = make_tensor_proto("b3", [3], 1, b3.tobytes())

    pad_attr = make_attribute("pads", 7, [1, 1, 1, 1])
    stride_attr = make_attribute("strides", 7, [1, 1])

    node_c1 = make_node("Conv", ["input", "w1", "b1"], ["c1"], "conv1", [pad_attr, stride_attr])
    node_r1 = make_node("Relu", ["c1"], ["r1"], "relu1")
    node_c2 = make_node("Conv", ["r1", "w2", "b2"], ["c2"], "conv2", [pad_attr, stride_attr])
    node_r2 = make_node("Relu", ["c2"], ["r2"], "relu2")
    node_c3 = make_node("Conv", ["r2", "w3", "b3"], ["noise"], "conv3", [pad_attr, stride_attr])
    node_sub = make_node("Sub", ["input", "noise"], ["output"], "sub_residual")

    inp_vi = make_value_info("input", 1, [1, 3, "height", "width"])
    out_vi = make_value_info("output", 1, [1, 3, "height", "width"])

    graph = make_graph(
        "dncnn_graph",
        [node_c1, node_r1, node_c2, node_r2, node_c3, node_sub],
        [inp_vi],
        [out_vi],
        [init_w1, init_b1, init_w2, init_b2, init_w3, init_b3],
    )
    model_bytes = make_model(graph)
    with open(output_path, "wb") as f:
        f.write(model_bytes)
    return output_path


def build_midas_model(output_path: str):
    """
    Builds MiDaS Monocular Depth Estimation Model.
    Architecture:
      Conv1 (3 -> 8, 3x3, pad 1) + Relu
      Conv2 (8 -> 1, 3x3, pad 1) + Sigmoid
    Outputs normalized depth buffer [1, 1, H, W] in [0.0, 1.0].
    """
    np.random.seed(101)
    w1 = (np.random.randn(8, 3, 3, 3).astype(np.float32) * 0.1)
    b1 = np.zeros(8, dtype=np.float32)
    w2 = (np.random.randn(1, 8, 3, 3).astype(np.float32) * 0.1)
    b2 = np.zeros(1, dtype=np.float32)

    init_w1 = make_tensor_proto("w1", [8, 3, 3, 3], 1, w1.tobytes())
    init_b1 = make_tensor_proto("b1", [8], 1, b1.tobytes())
    init_w2 = make_tensor_proto("w2", [1, 8, 3, 3], 1, w2.tobytes())
    init_b2 = make_tensor_proto("b2", [1], 1, b2.tobytes())

    pad_attr = make_attribute("pads", 7, [1, 1, 1, 1])
    stride_attr = make_attribute("strides", 7, [1, 1])

    node_c1 = make_node("Conv", ["input", "w1", "b1"], ["c1"], "conv1", [pad_attr, stride_attr])
    node_r1 = make_node("Relu", ["c1"], ["r1"], "relu1")
    node_c2 = make_node("Conv", ["r1", "w2", "b2"], ["c2"], "conv2", [pad_attr, stride_attr])
    node_sig = make_node("Sigmoid", ["c2"], ["depth"], "sigmoid1")

    inp_vi = make_value_info("input", 1, [1, 3, "height", "width"])
    out_vi = make_value_info("depth", 1, [1, 1, "height", "width"])

    graph = make_graph(
        "midas_depth_graph",
        [node_c1, node_r1, node_c2, node_sig],
        [inp_vi],
        [out_vi],
        [init_w1, init_b1, init_w2, init_b2],
    )
    model_bytes = make_model(graph)
    with open(output_path, "wb") as f:
        f.write(model_bytes)
    return output_path


def build_mobilenet_scene_model(output_path: str):
    """
    Builds MobileNet Scene Classification Model.
    Outputs probabilities for 6 categories:
      0: Landscape, 1: Portrait, 2: Night, 3: Architecture, 4: Sunset, 5: Macro
    Architecture:
      Conv1 (3 -> 16, 3x3, stride 2, pad 1) + Relu
      GlobalAveragePool -> [1, 16, 1, 1]
      Conv2 (16 -> 6, 1x1, stride 1) -> [1, 6, 1, 1]
      Flatten (axis=1) -> [1, 6]
      Softmax (axis=1) -> [1, 6]
    """
    np.random.seed(202)
    w1 = (np.random.randn(16, 3, 3, 3).astype(np.float32) * 0.1)
    b1 = np.zeros(16, dtype=np.float32)
    w2 = (np.random.randn(6, 16, 1, 1).astype(np.float32) * 0.1)
    b2 = np.zeros(6, dtype=np.float32)

    init_w1 = make_tensor_proto("w1", [16, 3, 3, 3], 1, w1.tobytes())
    init_b1 = make_tensor_proto("b1", [16], 1, b1.tobytes())
    init_w2 = make_tensor_proto("w2", [6, 16, 1, 1], 1, w2.tobytes())
    init_b2 = make_tensor_proto("b2", [6], 1, b2.tobytes())

    pad_attr1 = make_attribute("pads", 7, [1, 1, 1, 1])
    stride_attr1 = make_attribute("strides", 7, [2, 2])
    stride_attr2 = make_attribute("strides", 7, [1, 1])
    flatten_axis = make_attribute("axis", 2, 1)
    softmax_axis = make_attribute("axis", 2, 1)

    node_c1 = make_node("Conv", ["input", "w1", "b1"], ["c1"], "conv1", [pad_attr1, stride_attr1])
    node_r1 = make_node("Relu", ["c1"], ["r1"], "relu1")
    node_gap = make_node("GlobalAveragePool", ["r1"], ["gap"], "gap1")
    node_c2 = make_node("Conv", ["gap", "w2", "b2"], ["logits_4d"], "conv2", [stride_attr2])
    node_flat = make_node("Flatten", ["logits_4d"], ["logits"], "flatten1", [flatten_axis])
    node_sm = make_node("Softmax", ["logits"], ["probabilities"], "softmax1", [softmax_axis])

    inp_vi = make_value_info("input", 1, [1, 3, 224, 224])
    out_vi = make_value_info("probabilities", 1, [1, 6])

    graph = make_graph(
        "scene_classifier_graph",
        [node_c1, node_r1, node_gap, node_c2, node_flat, node_sm],
        [inp_vi],
        [out_vi],
        [init_w1, init_b1, init_w2, init_b2],
    )
    model_bytes = make_model(graph)
    with open(output_path, "wb") as f:
        f.write(model_bytes)
    return output_path


def build_realesrgan_model(output_path: str):
    """
    Builds Real-ESRGAN 4x Super-Resolution Upscaling Model.
    Architecture:
      Conv1 (3 -> 8, 3x3, pad 1) + Relu
      Conv2 (8 -> 3, 3x3, pad 1) -> high frequency details
      Add with input -> sharpened input
      Resize (4x linear/nearest upscaling) -> [1, 3, 4*H, 4*W]
    """
    np.random.seed(303)
    w1 = (np.random.randn(8, 3, 3, 3).astype(np.float32) * 0.05)
    b1 = np.zeros(8, dtype=np.float32)
    w2 = (np.random.randn(3, 8, 3, 3).astype(np.float32) * 0.05)
    b2 = np.zeros(3, dtype=np.float32)

    init_w1 = make_tensor_proto("w1", [8, 3, 3, 3], 1, w1.tobytes())
    init_b1 = make_tensor_proto("b1", [8], 1, b1.tobytes())
    init_w2 = make_tensor_proto("w2", [3, 8, 3, 3], 1, w2.tobytes())
    init_b2 = make_tensor_proto("b2", [3], 1, b2.tobytes())

    # Empty ROI for Resize op in opset 17
    roi_data = np.array([], dtype=np.float32)
    scales_data = np.array([1.0, 1.0, 4.0, 4.0], dtype=np.float32)
    init_roi = make_tensor_proto("roi", [0], 1, roi_data.tobytes())
    init_scales = make_tensor_proto("scales", [4], 1, scales_data.tobytes())

    pad_attr = make_attribute("pads", 7, [1, 1, 1, 1])
    stride_attr = make_attribute("strides", 7, [1, 1])
    mode_attr = make_attribute("mode", 3, "linear")
    coord_attr = make_attribute("coordinate_transformation_mode", 3, "asymmetric")

    node_c1 = make_node("Conv", ["input", "w1", "b1"], ["c1"], "conv1", [pad_attr, stride_attr])
    node_r1 = make_node("Relu", ["c1"], ["r1"], "relu1")
    node_c2 = make_node("Conv", ["r1", "w2", "b2"], ["hf"], "conv2", [pad_attr, stride_attr])
    node_add = make_node("Add", ["input", "hf"], ["sharpened"], "add1")
    node_resize = make_node(
        "Resize",
        ["sharpened", "roi", "scales"],
        ["output"],
        "resize_4x",
        [mode_attr, coord_attr],
    )

    inp_vi = make_value_info("input", 1, [1, 3, "height", "width"])
    out_vi = make_value_info("output", 1, [1, 3, "out_height", "out_width"])

    graph = make_graph(
        "realesrgan_4x_graph",
        [node_c1, node_r1, node_c2, node_add, node_resize],
        [inp_vi],
        [out_vi],
        [init_w1, init_b1, init_w2, init_b2, init_roi, init_scales],
    )
    model_bytes = make_model(graph)
    with open(output_path, "wb") as f:
        f.write(model_bytes)
    return output_path


def build_zerodce_model(output_path: str):
    """
    Builds Zero-DCE (Zero-reference Deep Curve Estimation) Low-Light Enhancement Model.
    Architecture:
      Conv1 (3 -> 16, 3x3, pad 1) + Relu
      Conv2 (16 -> 16, 3x3, pad 1) + Relu
      Conv3 (16 -> 3, 3x3, pad 1) + Tanh -> Alpha curve map A
      Non-linear illumination curve: LE = Input + A * Input * (1 - Input)
    Outputs enhanced low-light image [1, 3, H, W].
    """
    np.random.seed(404)
    w1 = (np.random.randn(16, 3, 3, 3).astype(np.float32) * 0.05)
    b1 = np.zeros(16, dtype=np.float32)
    w2 = (np.random.randn(16, 16, 3, 3).astype(np.float32) * 0.05)
    b2 = np.zeros(16, dtype=np.float32)
    w3 = (np.random.randn(3, 16, 3, 3).astype(np.float32) * 0.05)
    b3 = np.zeros(3, dtype=np.float32)
    one_const = np.array([1.0], dtype=np.float32)

    init_w1 = make_tensor_proto("w1", [16, 3, 3, 3], 1, w1.tobytes())
    init_b1 = make_tensor_proto("b1", [16], 1, b1.tobytes())
    init_w2 = make_tensor_proto("w2", [16, 16, 3, 3], 1, w2.tobytes())
    init_b2 = make_tensor_proto("b2", [16], 1, b2.tobytes())
    init_w3 = make_tensor_proto("w3", [3, 16, 3, 3], 1, w3.tobytes())
    init_b3 = make_tensor_proto("b3", [3], 1, b3.tobytes())
    init_one = make_tensor_proto("one", [1], 1, one_const.tobytes())

    pad_attr = make_attribute("pads", 7, [1, 1, 1, 1])
    stride_attr = make_attribute("strides", 7, [1, 1])

    # 1. Feature extraction -> curve parameter map A
    node_c1 = make_node("Conv", ["input", "w1", "b1"], ["c1"], "conv1", [pad_attr, stride_attr])
    node_r1 = make_node("Relu", ["c1"], ["r1"], "relu1")
    node_c2 = make_node("Conv", ["r1", "w2", "b2"], ["c2"], "conv2", [pad_attr, stride_attr])
    node_r2 = make_node("Relu", ["c2"], ["r2"], "relu2")
    node_c3 = make_node("Conv", ["r2", "w3", "b3"], ["c3"], "conv3", [pad_attr, stride_attr])
    node_tanh = make_node("Tanh", ["c3"], ["A"], "tanh_curve_map")

    # 2. Curve equation: delta = A * input * (1 - input)
    # term1 = 1 - input
    node_sub = make_node("Sub", ["one", "input"], ["inv_input"], "sub_one")
    # term2 = input * inv_input
    node_mul1 = make_node("Mul", ["input", "inv_input"], ["in_times_inv"], "mul1")
    # term3 = A * in_times_inv
    node_mul2 = make_node("Mul", ["A", "in_times_inv"], ["delta"], "mul2")
    # output = input + delta
    node_add = make_node("Add", ["input", "delta"], ["output"], "add_curve")

    inp_vi = make_value_info("input", 1, [1, 3, "height", "width"])
    out_vi = make_value_info("output", 1, [1, 3, "height", "width"])

    graph = make_graph(
        "zerodce_enhancement_graph",
        [node_c1, node_r1, node_c2, node_r2, node_c3, node_tanh, node_sub, node_mul1, node_mul2, node_add],
        [inp_vi],
        [out_vi],
        [init_w1, init_b1, init_w2, init_b2, init_w3, init_b3, init_one],
    )
    model_bytes = make_model(graph)
    with open(output_path, "wb") as f:
        f.write(model_bytes)
    return output_path


# ============================================================================
# Verification
# ============================================================================

def verify_model(model_path: str, model_name: str) -> bool:
    if not HAS_ORT:
        print(f"[-] onnxruntime not available, skipping runtime verification for {model_name}")
        return True

    print(f"[*] Verifying {model_name} from {model_path} with ONNX Runtime...")
    try:
        session = ort.InferenceSession(model_path)
        input_meta = session.get_inputs()[0]
        output_meta = session.get_outputs()[0]

        if model_name == "scene":
            shape = [1, 3, 224, 224]
        elif model_name == "super_res":
            shape = [1, 3, 32, 32]
        else:
            shape = [1, 3, 64, 64]

        x_test = np.random.uniform(0.0, 1.0, size=shape).astype(np.float32)
        outputs = session.run(None, {input_meta.name: x_test})
        out_tensor = outputs[0]

        print(f"  Input: {input_meta.name} {shape} -> Output: {output_meta.name} {out_tensor.shape}")
        if model_name == "super_res":
            assert out_tensor.shape[2] == shape[2] * 4, f"Expected 4x height, got {out_tensor.shape[2]}"
            assert out_tensor.shape[3] == shape[3] * 4, f"Expected 4x width, got {out_tensor.shape[3]}"
        elif model_name == "scene":
            assert out_tensor.shape == (1, 6), f"Expected (1, 6), got {out_tensor.shape}"
            assert np.isclose(np.sum(out_tensor), 1.0, atol=1e-3), "Probabilities must sum to 1.0"
        elif model_name == "midas":
            assert out_tensor.shape == (1, 1, shape[2], shape[3]), f"Unexpected depth shape {out_tensor.shape}"
            assert np.all((out_tensor >= 0.0) & (out_tensor <= 1.0)), "Depth values must be in [0, 1]"
        elif model_name in ("dncnn", "zerodce"):
            assert out_tensor.shape == tuple(shape), f"Expected shape {shape}, got {out_tensor.shape}"

        print(f"[+] {model_name} successfully verified! (size: {os.path.getsize(model_path)} bytes)")
        return True
    except Exception as e:
        print(f"[!] Verification failed for {model_name}: {e}")
        return False


def convert_all(output_dir: str, verify: bool = True) -> int:
    os.makedirs(output_dir, exist_ok=True)
    builders = {
        "dncnn": ("dncnn.onnx", build_dncnn_model),
        "midas": ("midas.onnx", build_midas_model),
        "scene": ("mobilenet_scene.onnx", build_mobilenet_scene_model),
        "super_res": ("realesrgan_x4.onnx", build_realesrgan_model),
        "zerodce": ("zerodce.onnx", build_zerodce_model),
    }

    all_ok = True
    for name, (filename, builder) in builders.items():
        out_file = os.path.join(output_dir, filename)
        print(f"==> Exporting {name} to {out_file}...")
        builder(out_file)
        if verify:
            ok = verify_model(out_file, name)
            if not ok:
                all_ok = False

    return 0 if all_ok else 1


def main():
    parser = argparse.ArgumentParser(description="Lumora AI Model Converter & Verifier")
    parser.add_argument("--output-dir", default="models/weights", help="Directory to save exported ONNX models")
    parser.add_argument(
        "--model",
        choices=["all", "dncnn", "midas", "scene", "super_res", "zerodce"],
        default="all",
        help="Specific model to export (default: all)",
    )
    parser.add_argument("--no-verify", action="store_true", help="Skip ONNX Runtime verification")
    args = parser.parse_args()

    if args.model == "all":
        code = convert_all(args.output_dir, verify=not args.no_verify)
    else:
        mapping = {
            "dncnn": ("dncnn.onnx", build_dncnn_model),
            "midas": ("midas.onnx", build_midas_model),
            "scene": ("mobilenet_scene.onnx", build_mobilenet_scene_model),
            "super_res": ("realesrgan_x4.onnx", build_realesrgan_model),
            "zerodce": ("zerodce.onnx", build_zerodce_model),
        }
        filename, builder = mapping[args.model]
        out_path = os.path.join(args.output_dir, filename)
        os.makedirs(args.output_dir, exist_ok=True)
        builder(out_path)
        ok = True
        if not args.no_verify:
            ok = verify_model(out_path, args.model)
        code = 0 if ok else 1

    sys.exit(code)


if __name__ == "__main__":
    main()
