#!/usr/bin/env python3
"""
Exact comparison - create a multi-head model that produces identical results
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

def create_exact_multihead_model(standard_model, device):
    """Create a multi-head model that produces EXACTLY the same results"""
    
    # Get the detection layer info
    detect_layer = standard_model.model[-1]
    
    print(f"Creating exact replica of {type(detect_layer)} with implicit layers")
    
    # Get input channels directly from the detection layer
    input_channels = [conv.in_channels for conv in detect_layer.m]
    
    # Convert anchors to the format expected by MultiHeadDetect
    anchors_list = []
    for i in range(detect_layer.nl):
        layer_anchors = detect_layer.anchors[i].flatten().tolist()
        anchors_list.append(layer_anchors)
    
    # Create single-head config that exactly matches the standard model
    head_configs = [
        {'name': 'general', 'classes': list(range(detect_layer.nc)), 'weight': 1.0}
    ]
    
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
    
    # Add implicit layers to MultiHeadDetect to match IDetect exactly
    from models.common import ImplicitA, ImplicitM
    
    # Add implicit layers for the general head
    multihead_detect.ia = torch.nn.ModuleList([ImplicitA(x) for x in input_channels])
    multihead_detect.im = torch.nn.ModuleList([ImplicitM(21) for _ in input_channels])  # 21 = (2+5)*3
    
    # Copy weights from standard detection layer
    for i, (std_conv, mh_conv) in enumerate(zip(detect_layer.m, multihead_detect.heads['general'])):
        mh_conv.weight.data.copy_(std_conv.weight.data)
        mh_conv.bias.data.copy_(std_conv.bias.data)
    
    # Copy implicit layer weights
    for i, (std_ia, mh_ia) in enumerate(zip(detect_layer.ia, multihead_detect.ia)):
        mh_ia.implicit.data.copy_(std_ia.implicit.data)
    
    for i, (std_im, mh_im) in enumerate(zip(detect_layer.im, multihead_detect.im)):
        mh_im.implicit.data.copy_(std_im.implicit.data)
    
    # Modify the forward method to use implicit layers like IDetect
    def forward_with_implicit(self, x):
        z = []  # inference output
        head_outputs = {}  # outputs from each head
        self.training |= self.export
        
        # Process the general head with implicit layers (like IDetect)
        head_config = self.head_configs[0]  # Only one head
        head_name = head_config['name']
        head_nc = len(head_config['classes'])
        head_no = head_nc + 5
        head_z = []
        
        for i in range(self.nl):
            # Apply implicit layers like IDetect does
            x_processed = self.ia[i](x[i])  # ImplicitA
            head_x = self.heads[head_name][i](x_processed)  # Conv
            head_x = self.im[i](head_x)  # ImplicitM
            
            bs, _, ny, nx = head_x.shape
            head_x = head_x.view(bs, self.na, head_no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            
            if not self.training:  # inference
                if self.grid[i].shape[2:4] != head_x.shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(head_x.device)
                
                y = head_x.sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                head_z.append(y.view(bs, -1, head_no))
        
        # Store head output
        if self.training:
            head_outputs[head_name] = [x[i] for i in range(self.nl)]
        else:
            head_outputs[head_name] = torch.cat(head_z, 1)
        
        # Return single head output (since we only have one)
        if self.training:
            return head_outputs
        else:
            return head_outputs[head_name]
    
    # Replace the forward method
    multihead_detect.forward = forward_with_implicit.__get__(multihead_detect, MultiHeadDetect)
    
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
    
    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres)
    
    return pred

def compare_tensors(tensor1, tensor2, name="tensors", tolerance=1e-6):
    """Compare two tensors with detailed output"""
    if tensor1.shape != tensor2.shape:
        print(f"❌ {name} shape mismatch: {tensor1.shape} vs {tensor2.shape}")
        return False
    
    max_diff = torch.max(torch.abs(tensor1 - tensor2)).item()
    mean_diff = torch.mean(torch.abs(tensor1 - tensor2)).item()
    
    if max_diff < tolerance:
        print(f"✅ {name} match within tolerance (max diff: {max_diff:.2e}, mean diff: {mean_diff:.2e})")
        return True
    else:
        print(f"❌ {name} differ (max diff: {max_diff:.6f}, mean diff: {mean_diff:.6f})")
        return False

def main():
    # Configuration
    weights_path = r"C:\Users\ben93\My Drive\Weights\flibs\flibs.pt"
    img_path = r"C:\Users\ben93\Downloads\fig03_hd.png"
    device = select_device('')
    img_size = 640
    
    print("🎯 Exact Comparison: Standard vs Multi-Head Detection")
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
        
        # Create exact multi-head model
        print("🔧 Creating exact multi-head model...")
        multihead_model = create_exact_multihead_model(standard_model, device)
        print(f"✅ Multi-head model created successfully")
        
        # Preprocess image
        print("🖼️  Preprocessing image...")
        img_tensor, img0 = preprocess_image(img_path, img_size)
        print(f"✅ Image preprocessed: {img_tensor.shape}")
        
        # Compare raw model outputs first
        print("\n🔍 Comparing raw model outputs (before NMS)...")
        with torch.no_grad():
            raw_standard = standard_model(img_tensor.to(device))[0]
            raw_multihead = multihead_model(img_tensor.to(device))
            
            print(f"Raw standard output shape: {raw_standard.shape}")
            print(f"Raw multi-head output shape: {raw_multihead.shape}")
            
            raw_match = compare_tensors(raw_standard, raw_multihead, "Raw outputs")
        
        if raw_match:
            print("🎉 Raw outputs match! Models are producing identical results.")
        else:
            print("⚠️  Raw outputs differ. Let's check post-NMS results anyway.")
        
        # Run inference with standard model
        print("\n🚀 Running inference with standard model...")
        start_time = time.time()
        pred_standard = run_inference(standard_model, img_tensor, device)
        standard_time = time.time() - start_time
        print(f"✅ Standard inference completed in {standard_time:.3f}s")
        std_detections = len(pred_standard[0]) if pred_standard[0] is not None else 0
        print(f"   Detections: {std_detections}")
        
        # Run inference with multi-head model
        print("🚀 Running inference with multi-head model...")
        start_time = time.time()
        pred_multihead = run_inference(multihead_model, img_tensor, device)
        multihead_time = time.time() - start_time
        print(f"✅ Multi-head inference completed in {multihead_time:.3f}s")
        mh_detections = len(pred_multihead[0]) if pred_multihead[0] is not None else 0
        print(f"   Detections: {mh_detections}")
        
        # Compare final detection results
        print("\n🔍 Comparing final detection results...")
        
        if std_detections == 0 and mh_detections == 0:
            print("✅ Both models produced no detections - PERFECT MATCH")
        elif std_detections == 0 or mh_detections == 0:
            print("❌ One model has detections, the other doesn't")
        elif std_detections != mh_detections:
            print(f"❌ Different number of detections: {std_detections} vs {mh_detections}")
        else:
            # Same number of detections, compare the actual values
            detection_match = compare_tensors(pred_standard[0], pred_multihead[0], "Final detections")
            if detection_match:
                print("🎉 PERFECT MATCH: Multi-head detection produces identical results!")
            else:
                print("⚠️  Detections differ slightly")
            
            # Print actual detection values for comparison
            print(f"\n📋 Detection Values Comparison:")
            print(f"Standard Model Detections ({std_detections} detections):")
            for i, det in enumerate(pred_standard[0]):
                x1, y1, x2, y2, conf, cls = det.tolist()
                print(f"  Detection {i+1}: bbox=[{x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}], conf={conf:.4f}, class={int(cls)}")
            
            print(f"\nMulti-Head Model Detections ({mh_detections} detections):")
            for i, det in enumerate(pred_multihead[0]):
                x1, y1, x2, y2, conf, cls = det.tolist()
                print(f"  Detection {i+1}: bbox=[{x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}], conf={conf:.4f}, class={int(cls)}")
            
            # Show exact differences
            if std_detections > 0:
                print(f"\n🔍 Exact Differences:")
                diff_tensor = torch.abs(pred_standard[0] - pred_multihead[0])
                max_diff_per_detection = torch.max(diff_tensor, dim=1)[0]
                for i, max_diff in enumerate(max_diff_per_detection):
                    print(f"  Detection {i+1} max difference: {max_diff.item():.2e}")
                
                print(f"\n📊 Overall Statistics:")
                print(f"  Maximum difference across all values: {torch.max(diff_tensor).item():.2e}")
                print(f"  Mean difference across all values: {torch.mean(diff_tensor).item():.2e}")
                print(f"  Standard deviation of differences: {torch.std(diff_tensor).item():.2e}")
        
        # Performance comparison
        print("\n📊 Performance Comparison:")
        print(f"Standard model:   {standard_time:.3f}s")
        print(f"Multi-head model: {multihead_time:.3f}s")
        print(f"Overhead:         {((multihead_time - standard_time) / standard_time * 100):+.1f}%")
        
        # Summary
        print("\n📋 Summary:")
        if raw_match and std_detections == mh_detections:
            print("✅ SUCCESS: Multi-head detection produces identical results to standard detection!")
            print("🎯 The multi-head architecture does not interfere with detection performance.")
        else:
            print("⚠️  Models produce different results due to architectural differences.")
            print("   This may be acceptable depending on the use case.")
        
    except Exception as e:
        print(f"❌ Error during comparison: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()