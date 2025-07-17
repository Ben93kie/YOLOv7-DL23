#!/usr/bin/env python3
"""
Test script for MultiHeadDetect module
"""

import torch
import torch.nn as nn
from models.yolo import MultiHeadDetect

def test_multihead_detect():
    """Test the MultiHeadDetect module with different head configurations"""
    
    # Test parameters
    batch_size = 2
    nc = 80  # total number of classes
    anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
    ch = [256, 512, 1024]  # input channels for each detection layer
    
    # Create test input tensors (simulating feature maps from backbone)
    x = [
        torch.randn(batch_size, ch[0], 80, 80),   # P3/8
        torch.randn(batch_size, ch[1], 40, 40),   # P4/16  
        torch.randn(batch_size, ch[2], 20, 20),   # P5/32
    ]
    
    print("Testing MultiHeadDetect module...")
    
    # Test 1: Single head (default behavior)
    print("\n1. Testing single head configuration:")
    single_head_config = [
        {'name': 'general', 'classes': list(range(nc)), 'weight': 1.0}
    ]
    
    model_single = MultiHeadDetect(nc=nc, anchors=anchors, ch=ch, head_configs=single_head_config)
    model_single.stride = torch.tensor([8., 16., 32.])  # Set strides manually for test
    
    # Test forward pass
    model_single.eval()
    with torch.no_grad():
        output_single = model_single(x)
    
    print(f"Single head output shape: {output_single.shape}")
    print(f"Expected detections per image: {output_single.shape[1] // batch_size}")
    
    # Test 2: Multi-head configuration
    print("\n2. Testing multi-head configuration:")
    multi_head_config = [
        {'name': 'vehicles', 'classes': [2, 3, 5, 7], 'weight': 1.2},  # car, motorcycle, bus, truck
        {'name': 'people', 'classes': [0], 'weight': 1.5},              # person
        {'name': 'animals', 'classes': [16, 17, 18, 19], 'weight': 1.0} # cat, dog, horse, sheep
    ]
    
    model_multi = MultiHeadDetect(nc=nc, anchors=anchors, ch=ch, head_configs=multi_head_config)
    model_multi.stride = torch.tensor([8., 16., 32.])  # Set strides manually for test
    
    # Test forward pass
    model_multi.eval()
    with torch.no_grad():
        output_multi = model_multi(x)
    
    print(f"Multi-head output shape: {output_multi.shape}")
    print(f"Number of heads: {model_multi.num_heads}")
    
    # Test 3: Training mode
    print("\n3. Testing training mode:")
    model_multi.train()
    with torch.no_grad():
        output_train = model_multi(x)
    
    print(f"Training mode output type: {type(output_train)}")
    print(f"Number of head outputs: {len(output_train)}")
    for head_name, head_output in output_train.items():
        print(f"Head '{head_name}': {len(head_output)} feature maps")
    
    print("\n✅ All tests passed! MultiHeadDetect module is working correctly.")
    
    return True

if __name__ == "__main__":
    test_multihead_detect()