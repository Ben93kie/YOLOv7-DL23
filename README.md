# YOLOv7 with Horizon Prediction Branch

This repository contains a modified version of the official YOLOv7 implementation ([YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://arxiv.org/abs/2207.02696)). The primary modification is the addition of a secondary prediction head designed to estimate horizon lines within an image, alongside the standard object detection task.

## Key Modifications and Features

### 1. Dual-Head Architecture

The core change involves altering the model architecture to support two distinct output branches:

*   **Detection Head:** The standard YOLOv7 detection head, outputting bounding boxes, objectness scores, and class probabilities.
*   **Horizon Head:** A new head designed to predict parameters representing a horizon line. In this implementation, it outputs **two parameters** per image, potentially representing slope and y-intercept, although the exact interpretation depends on the training process and loss function used.

### 2. Configuration (`cfg/training/yolov7-horizon-deploybased.yaml`)

A dedicated configuration file defines the modified architecture. This YAML file specifies:

*   The standard YOLOv7 backbone and neck structure.
*   The standard `Detect` layers for object detection, processing features from P3, P4, and P5 scales.
*   A new `HorizonDetect` layer module, defined in `models/yolo.py`. This layer typically takes features from one of the intermediate backbone/neck layers (e.g., P4 output) and processes them through convolutional layers to regress the horizon parameters.
*   The definition of the `HorizonDetect` module includes its input source (e.g., `[-1, 1, HorizonDetect, [nc]] # horizon head` indicates it takes input from the previous layer).

### 3. Code Implementation (`models/yolo.py`, `models/common.py`)

Several code changes were necessary to implement and support the dual-head architecture:

*   **`models/yolo.py`**:
    *   **`HorizonDetect` Class:** A new `nn.Module` class was created, similar to the original `Detect` class, but tailored for horizon parameter regression. It typically consists of one or more convolutional layers to reduce feature dimensions and produce the desired number of output parameters (2 in this case).
    *   **`Model.forward()` and `Model.forward_once()`:** Modified to handle the two output heads. The `forward_once` method now returns a tuple containing the outputs from both the detection head and the horizon head. The main `forward` method is adjusted to accommodate this tuple output, especially during tracing or exporting.
    *   **`parse_model()` Refactoring:** This function, responsible for building the model from the YAML configuration, was significantly refactored. The original implementation had issues correctly calculating the number of input channels (`c1`) for layers when using relative 'from' indices (e.g., `[-1, ...]`, `[-2, ...]`). This led to `RuntimeError`s, particularly in `RepConv` blocks where identity connections require matching input and output channels. The refactoring ensures accurate tracking and propagation of channel dimensions throughout the network construction, resolving these errors.
*   **`models/common.py`**:
    *   Minor adjustments or debugging print statements might have been added and subsequently removed during the development process.

### 4. Utility Scripts

Several scripts were created or modified to test, export, and verify the modified model:

*   **`test_horizon.py`**:
    *   **Purpose:** To verify the forward pass of the model architecture defined in the YAML configuration. It loads the config, builds the model (without weights), creates a dummy input tensor (or loads real images), performs inference, and prints the shapes of the resulting detection and horizon output tensors.
    *   **Features:** Includes argument parsing for batch size, image size, device, and image source path. Handles batch processing correctly by adjusting input tensor dimensions.
*   **`export_horizon.py`**:
    *   **Purpose:** To export the **architecture** (not trained weights) of the dual-output model to the ONNX format. This allows the model structure to be visualized or used in ONNX-compatible inference engines.
    *   **Features:** Loads the model from the YAML config. Creates a dummy input tensor matching the specified image size and batch size. Uses `torch.onnx.export` to perform the conversion, ensuring both output heads (`output_detections`, `output_horizon`) are correctly named in the ONNX graph. Includes options for enabling dynamic axes (for variable batch size or image dimensions) and applying ONNX simplification (`onnxsim`) for potentially better performance.
*   **`infer_onnx_horizon.py`**:
    *   **Purpose:** To load an exported ONNX model (`.onnx` file) and run inference on a sample image to verify that both output nodes exist and have the expected shapes.
    *   **Features:** Takes the ONNX model path and an image path as input. Uses the `onnxruntime` library to load the model and run inference. Preprocesses the input image (resizing, normalization). Checks the names and shapes of the output tensors against the expected names (`output_detections`, `output_horizon`) defined during the export process.

## Usage

1.  **Test Architecture Forward Pass:**
    ```bash
    # Test with default batch size 1 and dummy input
    python test_horizon.py

    # Test with batch size 4 using a real image, on GPU 0
    python test_horizon.py --batch 4 --img-size 640 --source path/to/your/image.jpg --device 0
    ```

2.  **Export Architecture to ONNX:**
    ```bash
    # Export with default settings (batch 1, static axes, opset 12, yolov7_horizon.onnx)
    python export_horizon.py --cfg cfg/training/yolov7-horizon-deploybased.yaml

    # Export with dynamic batch axis, simplification, and custom output name
    python export_horizon.py --cfg cfg/training/yolov7-horizon-deploybased.yaml --dynamic --simplify --batch-size 1 -f yolov7_horizon_dynamic.onnx
    ```

3.  **Verify ONNX Model Outputs:**
    ```bash
    # Run inference with the exported ONNX model
    python infer_onnx_horizon.py --model yolov7_horizon.onnx --image path/to/your/test_image.jpg
    ```

## Important Note on Training

The modifications in this repository focus primarily on **architecture definition, forward pass validation, and export**. The standard training scripts (`train.py`, `train_aux.py`) and loss computations (`compute_loss` in `utils/loss.py`) **have not** been updated to handle the dual-output nature of the model or to incorporate a suitable loss function for the horizon prediction task. Significant further development would be required to train this model effectively.

## Horizon Prediction Extension

This version of YOLOv7 has been extended to predict a horizon line in addition to object detections.

**Key Features:**

*   **Dual Output:** The model outputs both bounding box detections and a horizon line prediction (represented by two parameters: y-intercept and slope, normalized relative to image dimensions).
*   **Configuration:** A specific configuration file (`cfg/training/yolov7-horizon-deploybased.yaml`) defines the modified model architecture.
*   **Training Data:** Requires additional label information for the horizon line. For each image (`image_name.jpg`), a corresponding text file (`image_name.txt`) should contain the standard YOLO detection labels, followed by a line with two space-separated floating-point numbers representing the normalized horizon y-intercept and slope. If only detections are present for an image, the horizon line will be skipped for that sample during training.
*   **Loss Function:** The `ComputeLoss` class in `utils/loss.py` has been updated to include a Mean Squared Error (MSE) loss component for the horizon prediction.
*   **Training:** Use `train.py` with the horizon-specific configuration (`--cfg cfg/training/yolov7-horizon-deploybased.yaml`) and appropriately prepared data.
*   **Testing/Evaluation:**
    *   `test_horizon.py`: Tests the model architecture's forward pass with dummy data (no weights needed).
    *   `test_horizon_trained.py`: Evaluates a trained model checkpoint on a dataset, calculating detection metrics and horizon prediction Mean Squared Error (MSE) and Mean Absolute Error (MAE).
*   **ONNX Export:** Use `export_horizon.py` to export the dual-output model to ONNX format. The outputs will be named `output_detections` and `output_horizon`.
*   **ONNX Inference:** Use `infer_onnx_horizon.py` to run inference with the exported ONNX model and verify the outputs.

## Export
