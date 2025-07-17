# Multi-Head YOLO: Technical Documentation

## 🏗️ Architecture Deep Dive

### Core Components

#### 1. MultiHeadDetect Class Structure

```python
class MultiHeadDetect(nn.Module):
    def __init__(self, nc=80, anchors=(), ch=(), head_configs=None):
        # Core detection parameters
        self.nc = nc                    # Total number of classes
        self.nl = len(anchors)          # Number of detection layers (3 for YOLOv7)
        self.na = len(anchors[0]) // 2  # Number of anchors per layer (3 for YOLOv7)
        
        # Multi-head configuration
        self.head_configs = head_configs
        self.num_heads = len(head_configs)
        
        # Detection heads dictionary
        self.heads = nn.ModuleDict()    # Individual detection heads
        self.shared_conv = nn.ModuleList()  # Optional shared processing
```

#### 2. Individual Head Construction

Each detection head is constructed as follows:

```python
for head_config in head_configs:
    head_name = head_config['name']
    head_nc = len(head_config['classes'])  # Classes for this head
    head_no = head_nc + 5                  # bbox(4) + objectness(1) + classes
    
    # Special case: single head with all classes (backward compatibility)
    if len(head_configs) == 1 and len(head_config['classes']) == self.nc:
        head_no = self.nc + 5  # Use original class count
    
    # Create convolution layers for each detection scale
    head_convs = nn.ModuleList([
        nn.Conv2d(ch[i], head_no * self.na, 1) for i in range(self.nl)
    ])
    self.heads[head_name] = head_convs
```

**Key Points:**
- Each head has its own set of convolution layers (one per detection scale)
- Output channels = `(num_classes + 5) * num_anchors`
- Backward compatibility maintained for single-head configurations

#### 3. Forward Pass Architecture

```python
def forward(self, x):
    head_outputs = {}
    
    for head_config in self.head_configs:
        head_name = head_config['name']
        head_z = []
        
        for i in range(self.nl):  # For each detection scale
            # Apply head-specific convolution
            head_x = self.heads[head_name][i](x[i])
            
            # Reshape for YOLO format: [bs, na, no, ny, nx] -> [bs, na, ny, nx, no]
            bs, _, ny, nx = head_x.shape
            head_x = head_x.view(bs, self.na, head_no, ny, nx).permute(0, 1, 3, 4, 2)
            
            # Apply YOLO transformations (sigmoid, anchor scaling, etc.)
            if not self.training:
                y = self._apply_yolo_transforms(head_x, i)
                head_z.append(y.view(bs, -1, head_no))
        
        head_outputs[head_name] = torch.cat(head_z, 1) if not self.training else head_x
    
    return self._combine_head_outputs(head_outputs)
```

## 🔄 Model Loading and Conversion

### Standard Model Loading

```python
def load_standard_model(weights_path, device):
    """Standard YOLOv7 model loading"""
    ckpt = torch.load(weights_path, map_location=device)
    
    if 'model' in ckpt:
        model = ckpt['model'].float()  # Extract model from checkpoint
    else:
        model = ckpt  # Direct model weights
    
    model.to(device).eval()
    return model
```

### Multi-Head Model Creation from Standard

```python
def create_multihead_from_standard(standard_model, head_configs, device):
    """Convert standard model to multi-head"""
    
    # 1. Extract detection layer information
    detect_layer = standard_model.model[-1]  # Last layer (IDetect/Detect)
    
    # 2. Get architecture parameters
    input_channels = [conv.in_channels for conv in detect_layer.m]
    anchors = convert_anchors_format(detect_layer.anchors)
    
    # 3. Create new multi-head detection layer
    multihead_detect = MultiHeadDetect(
        nc=detect_layer.nc,
        anchors=anchors,
        ch=input_channels,
        head_configs=head_configs
    )
    
    # 4. Copy weights and properties
    copy_detection_weights(detect_layer, multihead_detect)
    copy_implicit_layers(detect_layer, multihead_detect)  # If IDetect
    copy_model_attributes(detect_layer, multihead_detect)
    
    # 5. Replace detection layer
    multihead_model = deepcopy(standard_model)
    multihead_model.model[-1] = multihead_detect
    
    return multihead_model
```

### Key Differences in Loading

| Aspect | Standard Model | Multi-Head Model |
|--------|---------------|------------------|
| **Detection Layer** | Single `IDetect`/`Detect` | `MultiHeadDetect` with multiple heads |
| **Weight Structure** | Single conv per scale | Multiple convs per head per scale |
| **Output Format** | Single tensor | Dict of head outputs (training) / Combined tensor (inference) |
| **Implicit Layers** | Built-in `ImplicitA`/`ImplicitM` | Manually added for compatibility |

## 🔧 Implicit Layer Handling

### Standard IDetect with Implicit Layers

```python
class IDetect(nn.Module):
    def __init__(self, nc=80, anchors=(), ch=()):
        # Standard convolutions
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)
        
        # Implicit layers (YOLOv7 specific)
        self.ia = nn.ModuleList(ImplicitA(x) for x in ch)      # Input processing
        self.im = nn.ModuleList(ImplicitM(self.no * self.na) for _ in ch)  # Output processing
    
    def forward(self, x):
        for i in range(self.nl):
            x[i] = self.m[i](self.ia[i](x[i]))  # ImplicitA -> Conv
            x[i] = self.im[i](x[i])             # ImplicitM
```

### Multi-Head with Implicit Layer Compatibility

```python
def add_implicit_layers_to_multihead(multihead_detect, standard_detect):
    """Add implicit layers for exact compatibility"""
    from models.common import ImplicitA, ImplicitM
    
    # Add implicit layers
    multihead_detect.ia = nn.ModuleList([ImplicitA(x) for x in input_channels])
    multihead_detect.im = nn.ModuleList([ImplicitM(output_channels) for _ in input_channels])
    
    # Copy weights from standard model
    for i, (std_ia, mh_ia) in enumerate(zip(standard_detect.ia, multihead_detect.ia)):
        mh_ia.implicit.data.copy_(std_ia.implicit.data)
    
    for i, (std_im, mh_im) in enumerate(zip(standard_detect.im, multihead_detect.im)):
        mh_im.implicit.data.copy_(std_im.implicit.data)
    
    # Modify forward pass to use implicit layers
    def forward_with_implicit(self, x):
        # Apply implicit processing like IDetect
        for i in range(self.nl):
            x_processed = self.ia[i](x[i])                    # ImplicitA
            head_x = self.heads['general'][i](x_processed)    # Conv
            head_x = self.im[i](head_x)                       # ImplicitM
            # ... rest of processing
    
    multihead_detect.forward = forward_with_implicit.__get__(multihead_detect, MultiHeadDetect)
```

## 📊 Output Format Differences

### Training Mode

**Standard Model:**
```python
# Returns feature maps for loss computation
output = [
    feature_map_scale1,  # [bs, na, no, ny1, nx1]
    feature_map_scale2,  # [bs, na, no, ny2, nx2]  
    feature_map_scale3   # [bs, na, no, ny3, nx3]
]
```

**Multi-Head Model:**
```python
# Returns dictionary of head outputs
output = {
    'vehicles': [feature_maps_for_vehicle_head],
    'people': [feature_maps_for_people_head],
    'general': [feature_maps_for_general_head]
}
```

### Inference Mode

**Standard Model:**
```python
# Single detection tensor
output = torch.tensor([bs, num_detections, 6])  # [x1, y1, x2, y2, conf, class]
```

**Multi-Head Model:**
```python
# Combined detection tensor (identical format)
output = torch.tensor([bs, num_detections, 6])  # Same format as standard
```

## 🎯 Head Configuration Examples

### 1. Single Head (Backward Compatibility)
```python
head_configs = [
    {'name': 'general', 'classes': list(range(80)), 'weight': 1.0}
]
# Result: Identical to standard YOLOv7
```

### 2. Vehicle Detection Specialization
```python
head_configs = [
    {'name': 'vehicles', 'classes': [2, 3, 5, 7, 8], 'weight': 1.2},      # Cars, bikes, trucks
    {'name': 'traffic', 'classes': [9, 10, 11, 12, 13], 'weight': 1.5},   # Traffic signs/lights
    {'name': 'other', 'classes': list(range(14, 80)), 'weight': 1.0}       # Everything else
]
```

### 3. Security/Surveillance Setup
```python
head_configs = [
    {'name': 'people', 'classes': [0], 'weight': 2.0},                     # High priority: people
    {'name': 'vehicles', 'classes': [2, 3, 5, 7], 'weight': 1.5},         # Medium: vehicles
    {'name': 'objects', 'classes': [24, 26, 28], 'weight': 1.8},          # High: bags/luggage
    {'name': 'background', 'classes': [i for i in range(80) if i not in [0,2,3,5,7,24,26,28]], 'weight': 0.5}
]
```

## 🔍 Weight Copying Process

### 1. Standard to Multi-Head Weight Transfer

```python
def copy_detection_weights(standard_detect, multihead_detect):
    """Copy weights from standard to multi-head detection"""
    
    # For single-head configuration (backward compatibility)
    if len(multihead_detect.head_configs) == 1:
        head_name = multihead_detect.head_configs[0]['name']
        
        # Direct weight copying
        for i, (std_conv, mh_conv) in enumerate(zip(standard_detect.m, multihead_detect.heads[head_name])):
            if std_conv.weight.shape == mh_conv.weight.shape:
                mh_conv.weight.data.copy_(std_conv.weight.data)
                mh_conv.bias.data.copy_(std_conv.bias.data)
            else:
                raise ValueError(f"Weight shape mismatch at layer {i}")
    
    # For multi-head configuration
    else:
        # More complex weight initialization/transfer logic
        initialize_multihead_weights(standard_detect, multihead_detect)
```

### 2. Anchor Format Conversion

```python
def convert_anchors_format(standard_anchors):
    """Convert IDetect anchor format to MultiHeadDetect format"""
    
    # IDetect format: torch.tensor([nl, na, 2])
    # MultiHeadDetect format: list of lists [[x1,y1,x2,y2,x3,y3], ...]
    
    anchors_list = []
    for i in range(standard_anchors.shape[0]):  # For each detection layer
        layer_anchors = standard_anchors[i].flatten().tolist()
        anchors_list.append(layer_anchors)
    
    return anchors_list
```

## 🚀 Performance Optimizations

### 1. Shared Feature Processing
```python
# Optional: Process features once, use for multiple heads
shared_features = []
for i in range(self.nl):
    shared_features.append(self.shared_conv[i](x[i]))

# Then apply each head to shared features
for head_config in self.head_configs:
    for i in range(self.nl):
        head_x = self.heads[head_name][i](shared_features[i])
```

### 2. Memory Optimization
```python
# Avoid storing all head outputs in training mode
if self.training:
    # Return generator instead of storing all outputs
    return {name: self._generate_head_output(name, x) for name in self.heads.keys()}
```

### 3. Inference Optimization
```python
# Early exit for single-head case
if len(self.head_configs) == 1:
    return self._single_head_forward(x)  # Optimized path
else:
    return self._multi_head_forward(x)   # Full multi-head processing
```

## 🧪 Validation and Testing

### Exact Compatibility Test
```python
def test_exact_compatibility():
    """Verify multi-head produces identical results to standard"""
    
    # Load same weights in both models
    standard_model = load_standard_model(weights_path)
    multihead_model = create_single_head_multihead(weights_path)
    
    # Test on same input
    with torch.no_grad():
        std_output = standard_model(test_input)
        mh_output = multihead_model(test_input)
    
    # Verify exact match
    assert torch.allclose(std_output, mh_output, atol=1e-6)
    print("✅ Exact compatibility verified")
```

This technical documentation provides the detailed architecture understanding needed to implement, modify, and extend the multi-head YOLO detection system.