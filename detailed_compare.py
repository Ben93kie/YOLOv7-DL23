#!/usr/bin/env python3
"""
Detailed comparison with better debugging
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

from models.yolo import Model, MultiHeadDetect
from utils.general import non_max_suppression, scale_coords, check_img_size
from utils.torch_utils import select_device
from utils.datasets import letterbox

def load_standard_model(weights_path, device):
    """Load standard model with weights directly from checkpoint"""
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

def create_equivalent_multihead_model(standard_model, device):
    """Create an equivalent multi-head model that should produce identical results"""
    
    # Get the detection layer info
    detect_layer = standard_model.model[-1]
    
    print(f"Standard detection layer type: {type(detect_layer)}")
    print(f"Number of classes: {detect_layer.nc}")
    print(f"Number of detection layers: {detect_layer.nl}")
    print(f"Number of anchors: {detect_layer.na}")
    print(f"Anchors shape: {detect_layer.anchors.shape}")
    print(f"Anchors: {detect_layer.anchors}")
    
    # Print shapes of detection layer weights
    if hasattr(detect_layer, 'm'):
        for i, conv in enumerate(detect_layer.m):
            print(f"Detection conv {i}: weight shape {conv.weight.shape}, bias shape {conv.bias.shape}")
    
    # Get input channels directly from the detection layer
    if hasattr(detect_layer, 'm') and len(detect_layer.m) > 0:
        input_channels = [conv.in_channels for conv in detect_layer.m]
        print(f"Input channels from detection layer: {input_channels}")
    else:
        # Fallback: try to trace back from the detection layer
        input_channels = []
        try:
            for i in range(len(standard_model.model) - 2, -1, -1):
                layer = standard_model.model[i]
                if hasattr(layer, 'conv'):
                    if hasattr(layer.conv, 'out_channels'):
                        input_channels.append(layer.conv.out_channels)
                    elif hasattr(layer.conv, 'weight'):
                        input_channels.append(layer.conv.weight.shape[0])
                elif hasattr(layer, 'weight'):
                    input_channels.append(layer.weight.shape[0])
                
                if len(input_channels) >= 3:
                    break
            
            # Take the last 3 and reverse
            input_channels = input_channels[:3][::-1]
            
        except:
            input_channels = [256, 512, 1024]  # Default
    
    print(f"Input channels for detection: {input_channels}")
    
    # Create single-head config that exactly matches the standard model
    # IMPORTANT: We need to ensure the multi-head model produces the same output dimensions
    head_configs = [
        {'name': 'general', 'classes': list(range(detect_layer.nc)), 'weight': 1.0}
    ]
    
    # Convert anchors to the format expected by MultiHeadDetect
    # IDetect uses shape [nl, na, 2], but MultiHeadDetect expects a list/tuple format
    anchors_list = []
    for i in range(detect_layer.nl):
        layer_anchors = detect_layer.anchors[i].flatten().tolist()  # Convert [na, 2] to flat list
        anchors_list.append(layer_anchors)
    
    print(f"Converted anchors: {anchors_list}")
    
    # Create new multi-head detection layer
    multihead_detect = MultiHeadDetect(
        nc=detect_layer.nc,
        anchors=anchors_list,
        ch=input_channels,
        head_configs=head_configs
    )
    
    # Copy the stride and other properties
    multihead_detect.stride = detect_layer.stride.clone()
    multihead_detect.anchors = detect_layer.anchors.clone()
    multihead_detect.anchor_grid = detect_layer.anchor_grid.clone()
    
    print(f"MultiHead detection layer created")
    print(f"General head conv shapes:")
    for i, conv in enumerate(multihead_detect.heads['general']):
        print(f"  Head conv {i}: weight shape {conv.weight.shape}, bias shape {conv.bias.shape}")
    
    # Copy weights carefully
    if hasattr(detect_layer, 'm'):
        for i, (std_conv, mh_conv) in enumerate(zip(detect_layer.m, multihead_detect.heads['general'])):
            print(f"Copying weights for layer {i}:")
            print(f"  Standard: {std_conv.weight.shape} -> MultiHead: {mh_conv.weight.shape}")
            
            if std_conv.weight.shape == mh_conv.weight.shape:
                mh_conv.weight.data.copy_(std_conv.weight.data)
                mh_conv.bias.data.copy_(std_conv.bias.data)
                print(f"  ✅ Copied successfully")
            else:
                print(f"  ❌ Shape mismatch!")
                return None
    
    # Check if the standard model has implicit layers
    if hasattr(detect_layer, 'ia') and hasattr(detect_layer, 'im'):
        print("⚠️  Standard model uses IDetect with implicit layers!")
        print("   MultiHeadDetect doesn't have implicit layers - this will cause differences")
        for i, (ia, im) in enumerate(zip(detect_layer.ia, detect_layer.im)):
            print(f"   Layer {i}: ImplicitA {ia.implicit.shape}, ImplicitM {im.implicit.shape}")
    
    # Initialize shared conv weights as identity transformation
    for i, shared_conv in enumerate(multihead_detect.shared_conv):
        in_ch = shared_conv.in_channels
        out_ch = shared_conv.out_channels
        
        print(f"Shared conv {i}: {in_ch} -> {out_ch}")
        
        # Initialize as identity for the diagonal
        with torch.no_grad():
            shared_conv.weight.zero_()
            min_ch = min(in_ch, out_ch)
            for j in range(min_ch):
                shared_conv.weight[j, j, 0, 0] = 1.0
            shared_conv.bias.zero_()
    
    # Copy additional attributes from the original detection layer
    if hasattr(detect_layer, 'f'):
        multihead_detect.f = detect_layer.f
    if hasattr(detect_layer, 'i'):
        multihead_detect.i = detect_layer.i
    if hasattr(detect_layer, 'type'):
        multihead_detect.type = detect_layer.type
    if hasattr(detect_layer, 'np'):
        multihead_detect.np = detect_layer.np
    
    # Replace the detection layer in the model
    multihead_model = deepcopy(standard_model)
    multihead_model.model[-1] = multihead_detect
    
    # Ensure the model is in evaluation mode
    multihead_model.eval()
    multihead_detect.eval()
    
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
        pred = model(img_tensor)
        
        # Handle different output formats
        if isinstance(pred, tuple):
            pred = pred[0]  # Take the first element if it's a tuple
        elif isinstance(pred, dict):
            # If it's a dict (training mode), combine outputs
            combined = []
            for head_name, head_output in pred.items():
                if isinstance(head_output, list):
                    # Convert list of feature maps to detection format
                    continue  # Skip for now
                else:
                    combined.append(head_output)
            if combined:
                pred = torch.cat(combined, dim=1)
            else:
                return None
    
    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres)
    
    return pred

def main():
    # Configuration
    weights_path = r"C:\Users\ben93\My Drive\Weights\flibs\flibs.pt"
    img_path = r"C:\Users\ben93\Downloads\fig03_hd.png"
    device = select_device('')
    img_size = 640
    
    print("🔍 Detailed Comparison: Standard vs Multi-Head Detection")
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
        standard_model = load_standard_model(weights_path, device)
        print(f"✅ Standard model loaded successfully")
        
        # Create multi-head model
        print("🔧 Creating equivalent multi-head model...")
        multihead_model = create_equivalent_multihead_model(standard_model, device)
        
        if multihead_model is None:
            print("❌ Failed to create multi-head model")
            return
            
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
        if pred_standard and pred_standard[0] is not None:
            print(f"   Detections: {len(pred_standard[0])}")
        else:
            print(f"   No detections")
        
        # Run inference with multi-head model
        print("🚀 Running inference with multi-head model...")
        start_time = time.time()
        pred_multihead = run_inference(multihead_model, img_tensor, device)
        multihead_time = time.time() - start_time
        print(f"✅ Multi-head inference completed in {multihead_time:.3f}s")
        if pred_multihead and pred_multihead[0] is not None:
            print(f"   Detections: {len(pred_multihead[0])}")
        else:
            print(f"   No detections")
        
        # Compare results
        print("🔍 Comparing detection results...")
        
        # Handle None cases properly
        std_has_detections = pred_standard and len(pred_standard) > 0 and pred_standard[0] is not None
        mh_has_detections = pred_multihead and len(pred_multihead) > 0 and pred_multihead[0] is not None
        
        if not std_has_detections and not mh_has_detections:
            print("✅ Both models produced no detections - MATCH")
        elif not std_has_detections or not mh_has_detections:
            print("❌ One model has detections, the other doesn't")
            if std_has_detections:
                print(f"   Standard model detections: {pred_standard[0].shape}")
            if mh_has_detections:
                print(f"   Multi-head model detections: {pred_multihead[0].shape}")
        else:
            # Compare shapes
            if pred_standard[0].shape == pred_multihead[0].shape:
                # Compare values
                max_diff = torch.max(torch.abs(pred_standard[0] - pred_multihead[0])).item()
                if max_diff < 1e-5:
                    print(f"✅ SUCCESS: Predictions match within tolerance (max diff: {max_diff:.2e})")
                    print("🎉 Multi-head detection produces identical results!")
                else:
                    print(f"❌ DIFFERENCE: Max difference is {max_diff:.6f}")
                    print("⚠️  Multi-head detection produces different results")
                    
                    # Show some sample values for debugging
                    print(f"Standard first detection: {pred_standard[0][0]}")
                    print(f"Multi-head first detection: {pred_multihead[0][0]}")
            else:
                print(f"❌ Shape mismatch: {pred_standard[0].shape} vs {pred_multihead[0].shape}")
        
        # Let's also compare the raw model outputs before NMS
        print("\n🔍 Comparing raw model outputs (before NMS)...")
        with torch.no_grad():
            raw_standard = standard_model(img_tensor.to(device))[0]
            raw_multihead = multihead_model(img_tensor.to(device))
            
            print(f"Raw standard output type: {type(raw_standard)}")
            print(f"Raw multi-head output type: {type(raw_multihead)}")
            
            # Handle different output formats
            if isinstance(raw_multihead, tuple):
                raw_multihead = raw_multihead[0]
                print(f"Multi-head is tuple, using first element: {raw_multihead.shape}")
            elif isinstance(raw_multihead, dict):
                print(f"Multi-head is dict with keys: {list(raw_multihead.keys())}")
                # Check what's in the dict
                for head_name, head_output in raw_multihead.items():
                    print(f"  {head_name}: {type(head_output)} - {head_output.shape if hasattr(head_output, 'shape') else 'no shape'}")
                
                # Try to get the general head output
                if 'general' in raw_multihead:
                    raw_multihead = raw_multihead['general']
                    print(f"Using general head output: {raw_multihead.shape}")
                else:
                    print("No general head found, cannot compare")
                    return
            
            print(f"Raw standard output shape: {raw_standard.shape}")
            print(f"Raw multi-head output shape: {raw_multihead.shape}")
            
            if raw_standard.shape == raw_multihead.shape:
                raw_max_diff = torch.max(torch.abs(raw_standard - raw_multihead)).item()
                print(f"Raw outputs max difference: {raw_max_diff:.6f}")
                if raw_max_diff < 1e-5:
                    print("✅ Raw outputs match - issue might be in post-processing")
                else:
                    print("❌ Raw outputs differ - issue is in the model itself")
            else:
                print(f"❌ Raw output shape mismatch: {raw_standard.shape} vs {raw_multihead.shape}")
        
        # Performance comparison
        print("\n📊 Performance Comparison:")
        print(f"Standard model:   {standard_time:.3f}s")
        print(f"Multi-head model: {multihead_time:.3f}s")
        print(f"Overhead:         {((multihead_time - standard_time) / standard_time * 100):+.1f}%")
        
    except Exception as e:
        print(f"❌ Error during comparison: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()