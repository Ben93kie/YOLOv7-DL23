#!/usr/bin/env python3
"""
Compare inference between standard detection and multi-head detection
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import sys
import time
from copy import deepcopy

# Add project root to path
sys.path.append('./')

from models.yolo import Model
from utils.general import non_max_suppression, scale_coords, check_img_size
from utils.torch_utils import select_device
from utils.datasets import letterbox

def load_model_with_weights(weights_path, device):
    """Load model with weights directly from checkpoint"""
    # Load weights
    ckpt = torch.load(weights_path, map_location=device)
    
    # Extract model directly from checkpoint
    if 'model' in ckpt:
        model = ckpt['model'].float()
    else:
        raise ValueError("No model found in checkpoint")
    
    model.to(device).eval()
    
    # Set model to inference mode
    for m in model.modules():
        if type(m) in [torch.nn.Hardswish, torch.nn.LeakyReLU, torch.nn.ReLU, torch.nn.ReLU6, torch.nn.SiLU]:
            m.inplace = True
    
    return model

def create_multihead_model_from_standard(standard_model, device):
    """Create a multi-head model by modifying the standard model"""
    # Get the model configuration
    yaml_config = deepcopy(standard_model.yaml)
    
    # Modify the last layer to use MultiHeadDetect
    # Find the detection layer in the model
    for i, layer_config in enumerate(yaml_config['backbone'] + yaml_config['head']):
        if isinstance(layer_config, list) and len(layer_config) >= 2:
            if 'Detect' in str(layer_config[1]) or 'IDetect' in str(layer_config[1]):
                # Replace with MultiHeadDetect
                layer_config[1] = 'MultiHeadDetect'
                # Add head configuration
                if len(layer_config) < 3:
                    layer_config.append([])
                # Add multi-head config as additional parameter
                layer_config.append({
                    'head_configs': [
                        {'name': 'general', 'classes': list(range(80)), 'weight': 1.0}
                    ]
                })
                break
    
    # Create new model with modified config
    multihead_model = Model(yaml_config)
    
    # Copy weights from standard model
    standard_state_dict = standard_model.state_dict()
    multihead_state_dict = multihead_model.state_dict()
    
    # Copy matching weights
    for name, param in standard_state_dict.items():
        if name in multihead_state_dict:
            if param.shape == multihead_state_dict[name].shape:
                multihead_state_dict[name].copy_(param)
            else:
                print(f"Shape mismatch for {name}: {param.shape} vs {multihead_state_dict[name].shape}")
    
    multihead_model.to(device).eval()
    return multihead_model

def create_simple_multihead_model(weights_path, device):
    """Create a simple multi-head model by manually constructing it"""
    from models.yolo import MultiHeadDetect
    
    # Load the standard model first to get architecture info
    standard_model = load_model_with_weights(weights_path, device)
    
    # Get the detection layer info
    detect_layer = standard_model.model[-1]
    
    # Get input channels for detection layer
    # Find the layers that feed into the detection layer
    input_channels = []
    for i in range(len(standard_model.model) - 1, -1, -1):
        layer = standard_model.model[i]
        if hasattr(layer, 'conv') and hasattr(layer.conv, 'out_channels'):
            input_channels.append(layer.conv.out_channels)
            if len(input_channels) == 3:  # We need 3 detection layers
                break
        elif hasattr(layer, 'out_channels'):
            input_channels.append(layer.out_channels)
            if len(input_channels) == 3:
                break
    
    # If we couldn't find channels, use default values
    if len(input_channels) < 3:
        input_channels = [256, 512, 1024]  # Default YOLOv7 channels
    else:
        input_channels = input_channels[::-1]  # Reverse to get correct order
    
    # Create single-head config that matches the standard model exactly
    head_configs = [
        {'name': 'general', 'classes': list(range(detect_layer.nc)), 'weight': 1.0}
    ]
    
    # Create new multi-head detection layer
    multihead_detect = MultiHeadDetect(
        nc=detect_layer.nc,
        anchors=detect_layer.anchors.clone(),
        ch=input_channels,
        head_configs=head_configs
    )
    
    # Copy the stride and other properties
    multihead_detect.stride = detect_layer.stride.clone()
    multihead_detect.anchors = detect_layer.anchors.clone()
    multihead_detect.anchor_grid = detect_layer.anchor_grid.clone()
    
    # Copy weights from standard detection layer to the general head
    if hasattr(detect_layer, 'm'):
        for i, (std_conv, mh_conv) in enumerate(zip(detect_layer.m, multihead_detect.heads['general'])):
            mh_conv.weight.data.copy_(std_conv.weight.data)
            mh_conv.bias.data.copy_(std_conv.bias.data)
    
    # Initialize shared conv weights as identity transformation
    for i, shared_conv in enumerate(multihead_detect.shared_conv):
        in_ch = shared_conv.in_channels
        out_ch = shared_conv.out_channels
        min_ch = min(in_ch, out_ch)
        
        # Initialize as identity for the diagonal
        with torch.no_grad():
            shared_conv.weight.zero_()
            for j in range(min_ch):
                shared_conv.weight[j, j, 0, 0] = 1.0
            shared_conv.bias.zero_()
    
    # Replace the detection layer in the model
    multihead_model = deepcopy(standard_model)
    multihead_model.model[-1] = multihead_detect
    
    return multihead_model

def preprocess_image(img_path, img_size=640):
    """Preprocess image for inference"""
    # Load image
    img0 = cv2.imread(str(img_path))
    assert img0 is not None, f'Image Not Found {img_path}'
    
    # Letterbox
    img = letterbox(img0, img_size, stride=32)[0]
    
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    
    # Normalize
    img = torch.from_numpy(img).float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    
    return img, img0

def run_inference(model, img_tensor, device, conf_thres=0.25, iou_thres=0.45):
    """Run inference on image"""
    img_tensor = img_tensor.to(device)
    
    # Inference
    with torch.no_grad():
        pred = model(img_tensor)[0]
    
    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres)
    
    return pred

def compare_detections(pred1, pred2, tolerance=1e-3):
    """Compare two sets of predictions"""
    if len(pred1) != len(pred2):
        return False, f"Different number of images: {len(pred1)} vs {len(pred2)}"
    
    for i, (p1, p2) in enumerate(zip(pred1, pred2)):
        if p1 is None and p2 is None:
            continue
        if p1 is None or p2 is None:
            return False, f"Image {i}: One prediction is None"
        
        if p1.shape != p2.shape:
            return False, f"Image {i}: Different shapes: {p1.shape} vs {p2.shape}"
        
        # Compare values with tolerance
        if not torch.allclose(p1, p2, atol=tolerance):
            max_diff = torch.max(torch.abs(p1 - p2)).item()
            return False, f"Image {i}: Values differ by max {max_diff:.6f}"
    
    return True, "Predictions match within tolerance"

def main():
    # Configuration
    weights_path = r"C:\Users\ben93\My Drive\Weights\flibs\flibs.pt"
    img_path = r"C:\Users\ben93\Downloads\fig03_hd.png"
    device = select_device('')
    img_size = 640
    
    print("🔍 Comparing Standard vs Multi-Head Detection Inference")
    print(f"Weights: {weights_path}")
    print(f"Image: {img_path}")
    print(f"Device: {device}")
    print("-" * 60)
    
    # Check if files exist
    if not Path(weights_path).exists():
        print(f"❌ Weights file not found: {weights_path}")
        return
    
    if not Path(img_path).exists():
        print(f"❌ Image file not found: {img_path}")
        return
    
    try:
        # Load standard model
        print("📥 Loading standard model...")
        standard_model = load_model_with_weights(weights_path, device)
        print(f"✅ Standard model loaded successfully")
        
        # Create multi-head model
        print("🔧 Creating multi-head model...")
        multihead_model = create_simple_multihead_model(weights_path, device)
        print(f"✅ Multi-head model created successfully")
        
        # Preprocess image
        print("🖼️  Preprocessing image...")
        img_tensor, img0 = preprocess_image(img_path, img_size)
        print(f"✅ Image preprocessed: {img_tensor.shape}")
        
        # Run inference with standard model
        print("🚀 Running inference with standard model...")
        start_time = time.time()
        pred_standard = run_inference(standard_model, img_tensor, device)
        standard_time = time.time() - start_time
        print(f"✅ Standard inference completed in {standard_time:.3f}s")
        print(f"   Detections: {len(pred_standard[0]) if pred_standard[0] is not None else 0}")
        
        # Run inference with multi-head model
        print("🚀 Running inference with multi-head model...")
        start_time = time.time()
        pred_multihead = run_inference(multihead_model, img_tensor, device)
        multihead_time = time.time() - start_time
        print(f"✅ Multi-head inference completed in {multihead_time:.3f}s")
        print(f"   Detections: {len(pred_multihead[0]) if pred_multihead[0] is not None else 0}")
        
        # Compare results
        print("🔍 Comparing detection results...")
        match, message = compare_detections(pred_standard, pred_multihead)
        
        if match:
            print(f"✅ SUCCESS: {message}")
            print("🎉 Multi-head detection produces identical results to standard detection!")
        else:
            print(f"❌ DIFFERENCE: {message}")
            print("⚠️  Multi-head detection produces different results")
        
        # Performance comparison
        print("\n📊 Performance Comparison:")
        print(f"Standard model:   {standard_time:.3f}s")
        print(f"Multi-head model: {multihead_time:.3f}s")
        print(f"Overhead:         {((multihead_time - standard_time) / standard_time * 100):+.1f}%")
        
        # Detailed detection info
        if pred_standard[0] is not None and pred_multihead[0] is not None:
            print("\n📋 Detection Details:")
            print(f"Standard detections shape:   {pred_standard[0].shape}")
            print(f"Multi-head detections shape: {pred_multihead[0].shape}")
            
            if pred_standard[0].shape[0] > 0:
                print(f"First detection (standard):   {pred_standard[0][0]}")
                print(f"First detection (multi-head): {pred_multihead[0][0]}")
        
    except Exception as e:
        print(f"❌ Error during comparison: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()