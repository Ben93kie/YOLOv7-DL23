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
from utils.nms_with_indices import non_max_suppression_with_indices
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
    
    # Create multi-head config with general, distance, and heading heads
    head_configs = [
        {'name': 'general', 'classes': list(range(detect_layer.nc)), 'weight': 1.0, 'output_size': detect_layer.nc + 5},
        {'name': 'distance', 'classes': [0], 'weight': 1.0, 'output_size': 1},  # Only distance value
        {'name': 'heading', 'classes': [0], 'weight': 1.0, 'output_size': 1}    # Only heading value
    ]
    
    # Create new multi-head detection layer
    multihead_detect = MultiHeadDetect(
        nc=detect_layer.nc,
        anchors=anchors_list,
        ch=input_channels,
        head_configs=head_configs
    )
    
    # Move the multi-head detect to the correct device immediately
    multihead_detect.to(device)
    
    # Copy the stride and other properties, ensuring they're on the correct device
    multihead_detect.stride = detect_layer.stride.clone().to(device)
    multihead_detect.anchors = detect_layer.anchors.clone().to(device)
    multihead_detect.anchor_grid = detect_layer.anchor_grid.clone().to(device)
    
    # Add implicit layers to MultiHeadDetect to match IDetect exactly
    from models.common import ImplicitA, ImplicitM
    
    # Create implicit layers and immediately move them to the correct device
    ia_layers = []
    im_layers = []
    
    for i, ch in enumerate(input_channels):
        # Create ImplicitA layer and move to device
        ia_layer = ImplicitA(ch)
        ia_layer.to(device)
        ia_layers.append(ia_layer)
        
        # Create ImplicitM layer and move to device
        im_layer = ImplicitM(21)  # 21 = (2+5)*3
        im_layer.to(device)
        im_layers.append(im_layer)
    
    multihead_detect.ia = torch.nn.ModuleList(ia_layers)
    multihead_detect.im = torch.nn.ModuleList(im_layers)
    
    # Manually create convolution layers with correct output sizes for each head
    # This overrides the default MultiHeadDetect behavior
    multihead_detect.heads = torch.nn.ModuleDict()
    
    for head_config in head_configs:
        head_name = head_config['name']
        head_output_size = head_config.get('output_size', len(head_config['classes']) + 5)
        
        # Create convolution layers for this head with correct output size
        head_convs = torch.nn.ModuleList()
        for ch in input_channels:
            conv = torch.nn.Conv2d(ch, head_output_size * multihead_detect.na, 1)
            head_convs.append(conv)
        
        multihead_detect.heads[head_name] = head_convs
        print(f"Created {head_name} head with output size {head_output_size} per anchor ({head_output_size * multihead_detect.na} total)")
    
    # Move all heads to device
    multihead_detect.heads.to(device)
    
    # Copy weights from standard detection layer to general head only
    for i, (std_conv, mh_conv) in enumerate(zip(detect_layer.m, multihead_detect.heads['general'])):
        mh_conv.weight.data.copy_(std_conv.weight.data)
        mh_conv.bias.data.copy_(std_conv.bias.data)
    
    # Distance and heading heads keep their random initialization (untrained)
    
    # Copy implicit layer weights and ensure they're on the correct device
    for i, (std_ia, mh_ia) in enumerate(zip(detect_layer.ia, multihead_detect.ia)):
        # Ensure both source and destination are on the same device
        std_implicit = std_ia.implicit.data.to(device)
        mh_ia.implicit.data.copy_(std_implicit)
        # Explicitly move the implicit layer to device
        mh_ia.to(device)
    
    for i, (std_im, mh_im) in enumerate(zip(detect_layer.im, multihead_detect.im)):
        # Ensure both source and destination are on the same device
        std_implicit = std_im.implicit.data.to(device)
        mh_im.implicit.data.copy_(std_implicit)
        # Explicitly move the implicit layer to device
        mh_im.to(device)
    
    # Modify the forward method to use implicit layers and process all heads
    def forward_with_implicit(self, x):
        z = []  # inference output
        head_outputs = {}  # outputs from each head
        self.training |= self.export
        
        # Get the device from input tensor
        input_device = x[0].device
        
        # Ensure all model components are on the same device as input
        self.to(input_device)
        
        # Process all heads
        for head_config in self.head_configs:
            head_name = head_config['name']
            head_nc = len(head_config['classes'])
            # Use output_size if specified, otherwise default to standard YOLO format
            head_no = head_config.get('output_size', head_nc + 5)
            head_z = []
            
            for i in range(self.nl):
                # Ensure input is on the correct device
                x_input = x[i].to(input_device)
                
                # Apply implicit layers for general head only (others use raw features)
                if head_name == 'general':
                    x_processed = self.ia[i](x_input)  # ImplicitA
                    head_x = self.heads[head_name][i](x_processed)  # Conv
                    head_x = self.im[i](head_x)  # ImplicitM
                else:
                    # Distance and heading heads use raw features (no implicit layers)
                    head_x = self.heads[head_name][i](x_input)  # Conv only
                
                bs, _, ny, nx = head_x.shape
                head_x = head_x.view(bs, self.na, head_no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
                
                if not self.training:  # inference
                    if self.grid[i].shape[2:4] != head_x.shape[2:4]:
                        self.grid[i] = self._make_grid(nx, ny).to(input_device)
                    
                    y = head_x.sigmoid()
                    
                    # Only apply YOLO post-processing to general head (which has bbox format)
                    if head_name == 'general' and head_no >= 5:
                        # Ensure all tensors are on the same device as the input
                        grid_i = self.grid[i].to(input_device)
                        stride_i = self.stride[i].to(input_device)
                        anchor_grid_i = self.anchor_grid[i].to(input_device)
                        
                        y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid_i) * stride_i  # xy
                        y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid_i  # wh
                    # For distance/heading heads, just use the raw sigmoid output
                    
                    head_z.append(y.view(bs, -1, head_no))
            
            # Store head output
            if self.training:
                head_outputs[head_name] = [x[i] for i in range(self.nl)]
            else:
                head_outputs[head_name] = torch.cat(head_z, 1)
        
        # Return all head outputs for multi-head processing
        if self.training:
            return head_outputs
        else:
            # For inference, return the general head for NMS compatibility, but store all outputs
            self.last_head_outputs = head_outputs  # Store for later access
            return head_outputs['general']
    
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
    
    # Ensure the entire model is on the correct device
    multihead_model.to(device)
    multihead_detect.to(device)
    
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

def run_inference(model, img_tensor, device, conf_thres=0.25, iou_thres=0.45, return_indices=False):
    """Run inference on image"""
    img_tensor = img_tensor.to(device)
    
    # Inference
    with torch.no_grad():
        pred = model(img_tensor)
        
        # Handle different output formats
        if isinstance(pred, tuple):
            pred = pred[0]  # Take the first element if it's a tuple
    
    # Apply NMS with optional index tracking
    if return_indices:
        pred, indices = non_max_suppression_with_indices(pred, conf_thres, iou_thres, return_indices=True)
        return pred, indices
    else:
        pred = non_max_suppression(pred, conf_thres, iou_thres)
        return pred

def extract_head_values_for_detections(model, img_tensor, detections, detection_indices, device):
    """Extract distance and heading values for specific detections using tracked indices"""
    if detections is None or len(detections) == 0:
        return [], []
    
    # Get the detection layer
    detect_layer = model.model[-1]
    if not hasattr(detect_layer, 'last_head_outputs'):
        return [], []
    
    head_outputs = detect_layer.last_head_outputs
    
    # Get distance and heading outputs
    distance_output = head_outputs.get('distance', None)
    heading_output = head_outputs.get('heading', None)
    
    if distance_output is None or heading_output is None:
        return [], []
    
    detection_distances = []
    detection_headings = []
    
    # Use the tracked indices to extract the correct head values
    for i, original_idx in enumerate(detection_indices):
        if original_idx >= 0 and original_idx < distance_output.shape[1]:
            # Extract the exact head values that correspond to this detection
            dist_val = distance_output[0, original_idx, 0].item()  # [batch, original_anchor_idx, value]
            head_val = heading_output[0, original_idx, 0].item()
            detection_distances.append(dist_val)
            detection_headings.append(head_val)
        else:
            # Fallback for invalid indices (e.g., autolabels)
            detection_distances.append(0.0)
            detection_headings.append(0.0)
    
    return detection_distances, detection_headings



def draw_detections(img, detections, img_shape, class_names=None, title="Detections"):
    """Draw bounding boxes on image"""
    if class_names is None:
        class_names = {0: 'Object_A', 1: 'Object_B'}
    
    # Colors for different classes
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    
    # Copy image
    img_with_boxes = img.copy()
    
    if detections is not None and len(detections) > 0:
        # Scale coordinates back to original image size
        detections[:, :4] = scale_coords(img_shape, detections[:, :4], img.shape).round()
        
        for i, det in enumerate(detections):
            x1, y1, x2, y2, conf, cls = det.tolist()
            x1, y1, x2, y2, cls = int(x1), int(y1), int(x2), int(y2), int(cls)
            
            # Choose color based on class
            color = colors[cls % len(colors)]
            
            # Draw bounding box
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            class_name = class_names.get(cls, f'Class_{cls}')
            label = f'{class_name}: {conf:.3f}'
            
            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Draw background rectangle for text
            cv2.rectangle(img_with_boxes, (x1, y1 - text_height - baseline - 5), 
                         (x1 + text_width, y1), color, -1)
            
            # Draw text
            cv2.putText(img_with_boxes, label, (x1, y1 - baseline - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Add title
    cv2.putText(img_with_boxes, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img_with_boxes, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
    
    return img_with_boxes

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
    # Configuration - make paths flexible for different environments
    weights_path = "/home/kiefer/Downloads/flibs.pt"  # Updated path
    img_path = r"C:\Users\kiefer\PycharmProjects\YOLOv7-DL23\detection_results\original_image.jpg"  # Updated path
    device = select_device('')
    img_size = 640
    
    # Fallback paths if the above don't exist
    fallback_weights = [
        r"C:\Users\ben93\My Drive\Weights\flibs\flibs.pt",
        r"G:\Meine Ablage\Weights\flibs\flibs.pt",
        "flibs.pt"
        "/home/kiefer/Downloads/flibs.pt"
    ]
    
    fallback_images = [
        r"C:\Users\kiefer\PycharmProjects\YOLOv7-DL23\detection_results\original_image.jpg",
        r"C:\Users\ben93\Downloads\fig03_hd.png",
        "detection_results/original_image.jpg"
    ]
    
    # Find existing weights file
    for w_path in fallback_weights:
        if Path(w_path).exists():
            weights_path = w_path
            break
    
    # Find existing image file
    for i_path in fallback_images:
        if Path(i_path).exists():
            img_path = i_path
            break
    
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
        
        # Run inference with multi-head model (with index tracking)
        print("🚀 Running inference with multi-head model...")
        start_time = time.time()
        pred_multihead, detection_indices = run_inference(multihead_model, img_tensor, device, return_indices=True)
        multihead_time = time.time() - start_time
        print(f"✅ Multi-head inference completed in {multihead_time:.3f}s")
        mh_detections = len(pred_multihead[0]) if pred_multihead[0] is not None else 0
        print(f"   Detections: {mh_detections}")
        if mh_detections > 0:
            print(f"   Detection indices: {detection_indices[0].tolist()}")
        
        # Extract outputs from all heads
        print("\n🔍 Multi-Head Outputs Analysis:")
        detect_layer = multihead_model.model[-1]
        if hasattr(detect_layer, 'last_head_outputs'):
            head_outputs = detect_layer.last_head_outputs
            
            for head_name, head_output in head_outputs.items():
                print(f"\n📊 {head_name.upper()} Head Output:")
                print(f"   Shape: {head_output.shape}")
                print(f"   Min value: {head_output.min().item():.6f}")
                print(f"   Max value: {head_output.max().item():.6f}")
                print(f"   Mean value: {head_output.mean().item():.6f}")
                print(f"   Std value: {head_output.std().item():.6f}")
                
                # Show some sample values from the head
                if head_output.numel() > 0:
                    # Get first detection's values for this head
                    sample_values = head_output[0, :5].flatten()  # First 5 values
                    print(f"   Sample values: {[f'{v:.4f}' for v in sample_values.tolist()]}")
                    
                    if head_name == 'distance':
                        print(f"   💡 Distance head produces untrained outputs (random-like values)")
                        print(f"      These would represent distance predictions after training")
                    elif head_name == 'heading':
                        print(f"   💡 Heading head produces untrained outputs (random-like values)")
                        print(f"      These would represent heading/angle predictions after training")
                    elif head_name == 'general':
                        print(f"   💡 General head uses trained weights (produces actual detections)")
        else:
            print("   ⚠️  Head outputs not stored - check forward method implementation")
        
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
            
            # Print actual detection values
            print(f"\n📋 Detection Values Comparison:")
            
            print(f"Standard Model Detections ({std_detections} detections):")
            for i, det in enumerate(pred_standard[0]):
                x1, y1, x2, y2, conf, cls = det.tolist()
                print(f"  Detection {i+1}: bbox=[{x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}], conf={conf:.4f}, class={int(cls)}")
            
            print(f"\nMulti-Head Model Detections ({mh_detections} detections):")
            
            # Extract corresponding distance and heading values for each detection
            detection_distances, detection_headings = extract_head_values_for_detections(
                multihead_model, img_tensor, pred_multihead[0], detection_indices[0], device
            )
            
            for i, det in enumerate(pred_multihead[0]):
                x1, y1, x2, y2, conf, cls = det.tolist()
                
                # Get corresponding distance and heading values
                distance_val = detection_distances[i] if i < len(detection_distances) else "N/A"
                heading_val = detection_headings[i] if i < len(detection_headings) else "N/A"
                
                print(f"  Detection {i+1}: bbox=[{x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}], conf={conf:.4f}, class={int(cls)}")
                print(f"                   distance={distance_val:.4f}, heading={heading_val:.4f} (from model heads)")
            
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
        
        # Generate and save visualization images
        print("\n🎨 Generating visualization images...")
        
        # Class names for visualization
        class_names = {0: 'Object_A', 1: 'Object_B'}
        
        # Draw detections on images
        img_standard = draw_detections(
            img0, 
            pred_standard[0] if pred_standard[0] is not None else None,
            img_tensor.shape[2:],  # Shape of processed image
            class_names,
            "Standard Model Detections"
        )
        
        img_multihead = draw_detections(
            img0, 
            pred_multihead[0] if pred_multihead[0] is not None else None,
            img_tensor.shape[2:],  # Shape of processed image
            class_names,
            "Multi-Head Model Detections"
        )
        
        # Save images
        output_dir = Path("detection_results")
        output_dir.mkdir(exist_ok=True)
        
        standard_output = output_dir / "standard_model_detections.jpg"
        multihead_output = output_dir / "multihead_model_detections.jpg"
        original_output = output_dir / "original_image.jpg"
        
        cv2.imwrite(str(standard_output), img_standard)
        cv2.imwrite(str(multihead_output), img_multihead)
        cv2.imwrite(str(original_output), img0)
        
        print(f"✅ Visualization images saved:")
        print(f"   Original: {original_output}")
        print(f"   Standard model: {standard_output}")
        print(f"   Multi-head model: {multihead_output}")
        
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