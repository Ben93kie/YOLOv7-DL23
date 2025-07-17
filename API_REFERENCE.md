# Multi-Head YOLO API Reference

## 📚 Core Classes

### MultiHeadDetect

The main multi-head detection module that replaces standard YOLO detection layers.

```python
class MultiHeadDetect(nn.Module):
    def __init__(self, nc=80, anchors=(), ch=(), head_configs=None)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nc` | `int` | `80` | Total number of classes in the dataset |
| `anchors` | `tuple/list` | `()` | Anchor boxes for each detection scale |
| `ch` | `tuple/list` | `()` | Input channels for each detection scale |
| `head_configs` | `list[dict]` | `None` | Configuration for each detection head |

#### Head Configuration Format

```python
head_config = {
    'name': str,        # Unique identifier for the head
    'classes': list,    # List of class indices this head handles
    'weight': float     # Weight/priority for this head (default: 1.0)
}
```

#### Example Usage

```python
# Single head (backward compatible)
single_head = MultiHeadDetect(
    nc=80,
    anchors=[[10,13, 16,30, 33,23], [30,61, 62,45, 59,119], [116,90, 156,198, 373,326]],
    ch=[256, 512, 1024],
    head_configs=[
        {'name': 'general', 'classes': list(range(80)), 'weight': 1.0}
    ]
)

# Multi-head configuration
multi_head = MultiHeadDetect(
    nc=80,
    anchors=[[10,13, 16,30, 33,23], [30,61, 62,45, 59,119], [116,90, 156,198, 373,326]],
    ch=[256, 512, 1024],
    head_configs=[
        {'name': 'vehicles', 'classes': [2, 3, 5, 7], 'weight': 1.2},
        {'name': 'people', 'classes': [0], 'weight': 1.5},
        {'name': 'objects', 'classes': [16, 17, 18, 19], 'weight': 1.0}
    ]
)
```

#### Methods

##### `forward(x)`

Forward pass through the multi-head detection module.

**Parameters:**
- `x` (`list[torch.Tensor]`): List of feature maps from backbone, one per detection scale

**Returns:**
- **Training mode**: `dict` - Dictionary mapping head names to their feature outputs
- **Inference mode**: `torch.Tensor` - Combined detection tensor `[batch, detections, 6]`

**Example:**
```python
# Input: list of 3 feature maps
x = [
    torch.randn(1, 256, 80, 80),   # P3/8
    torch.randn(1, 512, 40, 40),   # P4/16
    torch.randn(1, 1024, 20, 20)   # P5/32
]

# Forward pass
model.eval()  # Inference mode
detections = model(x)  # Shape: [1, N, 6] where N is number of detections

model.train()  # Training mode
head_outputs = model(x)  # Dict: {'head1': features, 'head2': features, ...}
```

##### `_combine_head_outputs(head_outputs)`

Combines outputs from multiple heads into a single detection tensor.

**Parameters:**
- `head_outputs` (`dict`): Dictionary of head outputs

**Returns:**
- `torch.Tensor`: Combined detection tensor

##### `_make_grid(nx, ny)`

Creates coordinate grids for YOLO detection transformations.

**Parameters:**
- `nx` (`int`): Grid width
- `ny` (`int`): Grid height

**Returns:**
- `torch.Tensor`: Coordinate grid tensor

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `nc` | `int` | Number of classes |
| `nl` | `int` | Number of detection layers (scales) |
| `na` | `int` | Number of anchors per layer |
| `head_configs` | `list` | Head configuration list |
| `num_heads` | `int` | Number of detection heads |
| `heads` | `nn.ModuleDict` | Dictionary of detection head modules |
| `anchors` | `torch.Tensor` | Anchor tensors |
| `anchor_grid` | `torch.Tensor` | Anchor grid tensors |
| `stride` | `torch.Tensor` | Detection strides |
| `grid` | `list` | Coordinate grids |

## 🛠️ Utility Functions

### Model Loading and Conversion

#### `load_standard_model(weights_path, device)`

Load a standard YOLOv7 model from checkpoint.

```python
def load_standard_model(weights_path, device):
    """
    Load standard YOLOv7 model from weights file.
    
    Args:
        weights_path (str): Path to model weights (.pt file)
        device (torch.device): Target device for model
    
    Returns:
        torch.nn.Module: Loaded YOLOv7 model
    """
```

**Example:**
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_standard_model('yolov7.pt', device)
```

#### `create_multihead_from_standard(standard_model, head_configs, device)`

Convert a standard YOLOv7 model to multi-head architecture.

```python
def create_multihead_from_standard(standard_model, head_configs, device):
    """
    Convert standard model to multi-head architecture.
    
    Args:
        standard_model: Standard YOLOv7 model
        head_configs (list): List of head configurations
        device: Target device
    
    Returns:
        torch.nn.Module: Multi-head YOLOv7 model
    """
```

**Example:**
```python
head_configs = [
    {'name': 'vehicles', 'classes': [2, 3, 5, 7], 'weight': 1.2},
    {'name': 'people', 'classes': [0], 'weight': 1.5}
]

multihead_model = create_multihead_from_standard(
    standard_model, head_configs, device
)
```

### Preprocessing and Inference

#### `preprocess_image(img_path, img_size=640)`

Preprocess image for YOLO inference.

```python
def preprocess_image(img_path, img_size=640):
    """
    Preprocess image for YOLO inference.
    
    Args:
        img_path (str): Path to input image
        img_size (int): Target image size
    
    Returns:
        tuple: (processed_tensor, original_image, processed_image)
    """
```

#### `run_inference(model, img_tensor, device, conf_thres=0.25, iou_thres=0.45)`

Run inference on preprocessed image.

```python
def run_inference(model, img_tensor, device, conf_thres=0.25, iou_thres=0.45):
    """
    Run inference with NMS post-processing.
    
    Args:
        model: YOLO model (standard or multi-head)
        img_tensor: Preprocessed image tensor
        device: Computation device
        conf_thres (float): Confidence threshold
        iou_thres (float): IoU threshold for NMS
    
    Returns:
        list: List of detection tensors (one per image in batch)
    """
```

### Visualization

#### `draw_detections(img, detections, img_shape, class_names=None, title="Detections")`

Draw bounding boxes on image.

```python
def draw_detections(img, detections, img_shape, class_names=None, title="Detections"):
    """
    Draw detection bounding boxes on image.
    
    Args:
        img (np.ndarray): Input image (BGR format)
        detections (torch.Tensor): Detection tensor [N, 6]
        img_shape (tuple): Shape of processed image for coordinate scaling
        class_names (dict): Mapping of class indices to names
        title (str): Title to display on image
    
    Returns:
        np.ndarray: Image with drawn bounding boxes
    """
```

## 📋 Configuration Examples

### Predefined Configurations

#### Vehicle Detection
```python
VEHICLE_CONFIG = [
    {'name': 'cars', 'classes': [2], 'weight': 1.2},
    {'name': 'trucks', 'classes': [7], 'weight': 1.1},
    {'name': 'motorcycles', 'classes': [3], 'weight': 1.3},
    {'name': 'other', 'classes': [i for i in range(80) if i not in [2, 3, 7]], 'weight': 1.0}
]
```

#### Security/Surveillance
```python
SECURITY_CONFIG = [
    {'name': 'people', 'classes': [0], 'weight': 2.0},
    {'name': 'vehicles', 'classes': [2, 3, 5, 7], 'weight': 1.5},
    {'name': 'bags', 'classes': [24, 26, 28], 'weight': 1.8},
    {'name': 'other', 'classes': [i for i in range(80) if i not in [0,2,3,5,7,24,26,28]], 'weight': 0.8}
]
```

#### Indoor Scene
```python
INDOOR_CONFIG = [
    {'name': 'people_pets', 'classes': [0, 16, 17], 'weight': 1.3},
    {'name': 'furniture', 'classes': [56, 57, 58, 59, 60, 61, 62], 'weight': 1.1},
    {'name': 'electronics', 'classes': [63, 64, 65, 66, 67, 68], 'weight': 1.0},
    {'name': 'other', 'classes': [i for i in range(80) if i not in [0,16,17,56,57,58,59,60,61,62,63,64,65,66,67,68]], 'weight': 0.9}
]
```

## 🔧 Advanced Usage

### Custom Head Implementation

```python
class CustomDetectionHead(nn.Module):
    def __init__(self, in_channels, num_classes, num_anchors):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, (num_classes + 5) * num_anchors, 1)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        x = self.activation(self.conv1(x))
        return self.conv2(x)

# Integration with MultiHeadDetect
class CustomMultiHeadDetect(MultiHeadDetect):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Replace standard heads with custom heads
        for head_name, head_convs in self.heads.items():
            for i, conv in enumerate(head_convs):
                in_ch = conv.in_channels
                out_ch = conv.out_channels
                num_anchors = self.na
                num_classes = out_ch // num_anchors - 5
                
                self.heads[head_name][i] = CustomDetectionHead(
                    in_ch, num_classes, num_anchors
                )
```

### Training Integration

```python
def compute_multihead_loss(predictions, targets, head_configs):
    """
    Compute loss for multi-head training.
    
    Args:
        predictions (dict): Head predictions from MultiHeadDetect
        targets: Ground truth targets
        head_configs: Head configuration list
    
    Returns:
        dict: Loss components for each head
    """
    losses = {}
    
    for head_config in head_configs:
        head_name = head_config['name']
        head_classes = head_config['classes']
        head_weight = head_config['weight']
        
        # Filter targets for this head's classes
        head_targets = filter_targets_by_classes(targets, head_classes)
        
        # Compute loss for this head
        head_loss = compute_yolo_loss(predictions[head_name], head_targets)
        losses[head_name] = head_loss * head_weight
    
    return losses
```

## 🚨 Error Handling

### Common Exceptions

#### `ValueError: Weight shape mismatch`
Occurs when trying to load incompatible weights.
```python
try:
    multihead_model = create_multihead_from_standard(standard_model, head_configs)
except ValueError as e:
    print(f"Weight compatibility error: {e}")
    # Check head configurations and model architecture
```

#### `RuntimeError: Anchor format incompatible`
Occurs when anchor formats don't match.
```python
try:
    anchors_list = convert_anchors_format(detect_layer.anchors)
except RuntimeError as e:
    print(f"Anchor conversion error: {e}")
    # Verify anchor tensor format
```

### Debugging Utilities

```python
def debug_multihead_model(model, input_tensor):
    """Debug multi-head model forward pass."""
    print(f"Model type: {type(model.model[-1])}")
    print(f"Number of heads: {model.model[-1].num_heads}")
    
    # Hook to capture intermediate outputs
    def hook_fn(module, input, output):
        print(f"Layer output shape: {output.shape if hasattr(output, 'shape') else type(output)}")
    
    hooks = []
    for name, module in model.named_modules():
        if 'heads' in name:
            hooks.append(module.register_forward_hook(hook_fn))
    
    try:
        output = model(input_tensor)
        print(f"Final output shape: {output.shape}")
    finally:
        for hook in hooks:
            hook.remove()
```

## 📊 Performance Monitoring

### Benchmarking

```python
def benchmark_models(standard_model, multihead_model, test_data, num_runs=100):
    """
    Benchmark standard vs multi-head model performance.
    
    Args:
        standard_model: Standard YOLOv7 model
        multihead_model: Multi-head YOLOv7 model
        test_data: Test dataset
        num_runs: Number of benchmark runs
    
    Returns:
        dict: Performance metrics
    """
    import time
    
    # Warm up
    for _ in range(10):
        _ = standard_model(test_data)
        _ = multihead_model(test_data)
    
    # Benchmark standard model
    start_time = time.time()
    for _ in range(num_runs):
        _ = standard_model(test_data)
    standard_time = time.time() - start_time
    
    # Benchmark multi-head model
    start_time = time.time()
    for _ in range(num_runs):
        _ = multihead_model(test_data)
    multihead_time = time.time() - start_time
    
    return {
        'standard_avg_time': standard_time / num_runs,
        'multihead_avg_time': multihead_time / num_runs,
        'overhead_percent': ((multihead_time - standard_time) / standard_time) * 100
    }
```

This API reference provides comprehensive documentation for using and extending the multi-head YOLO detection system.