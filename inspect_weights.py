#!/usr/bin/env python3
"""
Inspect the weights file to understand its structure
"""

import torch
import sys

def inspect_weights(weights_path):
    """Inspect the structure of the weights file"""
    print(f"🔍 Inspecting weights: {weights_path}")
    
    try:
        # Load weights
        ckpt = torch.load(weights_path, map_location='cpu')
        
        print(f"✅ Weights loaded successfully")
        print(f"Keys in checkpoint: {list(ckpt.keys())}")
        
        if 'model' in ckpt:
            model_state = ckpt['model']
            if hasattr(model_state, 'yaml'):
                print(f"Model YAML config found")
                print(f"Number of classes: {model_state.yaml.get('nc', 'Unknown')}")
            
            state_dict = model_state.float().state_dict() if hasattr(model_state, 'state_dict') else ckpt
        else:
            state_dict = ckpt
        
        print(f"\n📊 Model structure analysis:")
        
        # Find detection layer weights
        detection_layers = []
        for name, param in state_dict.items():
            if 'model.105' in name or 'detect' in name.lower() or name.endswith('.m.0.weight') or name.endswith('.m.1.weight') or name.endswith('.m.2.weight'):
                detection_layers.append((name, param.shape))
        
        print(f"Detection layer weights:")
        for name, shape in detection_layers:
            print(f"  {name}: {shape}")
        
        # Analyze the output dimensions
        if detection_layers:
            # Find the main detection weight (not anchors)
            main_det_weights = [item for item in detection_layers if '.m.' in item[0] and '.weight' in item[0]]
            if main_det_weights:
                first_det_weight = main_det_weights[0][1]
                output_channels = first_det_weight[0]
                
                # Calculate number of classes
                # YOLO output format: (classes + 5) * num_anchors
                # Assuming 3 anchors per layer
                num_anchors = 3
                classes_plus_5 = output_channels // num_anchors
                num_classes = classes_plus_5 - 5
            else:
                num_classes = None
            
            print(f"\n📈 Calculated model parameters:")
            print(f"Output channels: {output_channels}")
            print(f"Classes + 5: {classes_plus_5}")
            print(f"Number of classes: {num_classes}")
            print(f"Number of anchors: {num_anchors}")
        
        # Check for other metadata
        if 'epoch' in ckpt:
            print(f"Epoch: {ckpt['epoch']}")
        if 'best_fitness' in ckpt:
            print(f"Best fitness: {ckpt['best_fitness']}")
        
        return num_classes if detection_layers else None
        
    except Exception as e:
        print(f"❌ Error inspecting weights: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    weights_path = r"C:\Users\ben93\My Drive\Weights\flibs\flibs.pt"
    num_classes = inspect_weights(weights_path)
    print(f"\n🎯 Detected number of classes: {num_classes}")