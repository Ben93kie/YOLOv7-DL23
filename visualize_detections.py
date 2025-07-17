#!/usr/bin/env python3
"""
Visualize detections by drawing bounding boxes on the image
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
    img_tensor = torch.from_numpy(img).float()
    img_tensor /= 255.0
    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.unsqueeze(0)
    
    return img_tensor, img0, img  # Return both original and processed

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

def draw_detections(img, detections, img_shape, class_names=None, title="Detections"):
    """Draw bounding boxes on image"""
    if class_names is None:
        class_names = {0: 'Class_0', 1: 'Class_1'}
    
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
            cv2.putText(img_with_boxes, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Add title
    cv2.putText(img_with_boxes, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img_with_boxes, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
    
    return img_with_boxes

def main():
    # Configuration
    weights_path = r"C:\Users\ben93\My Drive\Weights\flibs\flibs.pt"
    img_path = r"C:\Users\ben93\Downloads\fig03_hd.png"
    device = select_device('')
    img_size = 640
    
    print("🎯 Visualizing Detections: Standard vs Multi-Head")
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
        img_tensor, img0, img_processed = preprocess_image(img_path, img_size)
        print(f"✅ Image preprocessed: {img_tensor.shape}")
        print(f"   Original image shape: {img0.shape}")
        print(f"   Processed image shape: {img_processed.shape}")
        
        # Run inference with standard model
        print("\n🚀 Running inference with standard model...")
        pred_standard = run_inference(standard_model, img_tensor, device)
        std_detections = len(pred_standard[0]) if pred_standard[0] is not None else 0
        print(f"✅ Standard model found {std_detections} detections")
        
        # Run inference with multi-head model
        print("🚀 Running inference with multi-head model...")
        pred_multihead = run_inference(multihead_model, img_tensor, device)
        mh_detections = len(pred_multihead[0]) if pred_multihead[0] is not None else 0
        print(f"✅ Multi-head model found {mh_detections} detections")
        
        # Class names (you can customize these based on your model)
        class_names = {0: 'Object_A', 1: 'Object_B'}
        
        # Draw detections on images
        print("\n🎨 Drawing detections...")
        
        # Standard model visualization
        img_standard = draw_detections(
            img0, 
            pred_standard[0] if pred_standard[0] is not None else None,
            img_tensor.shape[2:],  # Shape of processed image
            class_names,
            "Standard Model Detections"
        )
        
        # Multi-head model visualization
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
        
        print(f"✅ Images saved:")
        print(f"   Original: {original_output}")
        print(f"   Standard model: {standard_output}")
        print(f"   Multi-head model: {multihead_output}")
        
        # Print detection details
        if std_detections > 0:
            print(f"\n📋 Detection Details:")
            print(f"Standard Model Detections ({std_detections}):")
            for i, det in enumerate(pred_standard[0]):
                x1, y1, x2, y2, conf, cls = det.tolist()
                class_name = class_names.get(int(cls), f'Class_{int(cls)}')
                print(f"  {i+1}. {class_name}: bbox=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}], conf={conf:.3f}")
            
            print(f"\nMulti-Head Model Detections ({mh_detections}):")
            for i, det in enumerate(pred_multihead[0]):
                x1, y1, x2, y2, conf, cls = det.tolist()
                class_name = class_names.get(int(cls), f'Class_{int(cls)}')
                print(f"  {i+1}. {class_name}: bbox=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}], conf={conf:.3f}")
        
        print(f"\n🎉 Visualization complete! Check the 'detection_results' folder.")
        
    except Exception as e:
        print(f"❌ Error during visualization: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()