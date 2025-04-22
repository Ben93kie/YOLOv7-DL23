"""Runs inference using an ONNX YOLOv7+Horizon model and checks outputs."""

import cv2
import numpy as np
import onnxruntime
import argparse
from pathlib import Path

def preprocess_image(image_path, img_size=(640, 640)):
    """Loads and preprocesses a single image for YOLOv7 ONNX inference."""
    print(f"Loading image: {image_path}")
    img0 = cv2.imread(str(image_path))
    if img0 is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    print(f"Preprocessing image (resize to {img_size}, BGR->RGB, HWC->CHW, normalize)...")
    # Resize
    img = cv2.resize(img0, img_size)
    # BGR to RGB, HWC to CHW
    img = img[:, :, ::-1].transpose(2, 0, 1)
    # Make contiguous
    img = np.ascontiguousarray(img)
    # Convert to float32 and normalize
    img = img.astype(np.float32) / 255.0
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    print(f"Preprocessed image shape: {img.shape}")
    return img

def main(opt):
    print(f"Loading ONNX model: {opt.model}")
    try:
        session = onnxruntime.InferenceSession(str(opt.model), providers=['CPUExecutionProvider']) # Use CPU for simplicity
        print("ONNX model loaded successfully.")
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        return

    # Get model input/output details
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    print(f"  Model Input Name: {input_name}")
    print(f"  Model Output Names: {output_names}")

    # Preprocess image
    try:
        input_image = preprocess_image(opt.image, opt.img_size)
    except Exception as e:
        print(f"Error during image preprocessing: {e}")
        return

    # Run inference
    print("Running ONNX inference...")
    try:
        outputs = session.run(output_names, {input_name: input_image})
        print("Inference complete.")
    except Exception as e:
        print(f"Error during ONNX inference: {e}")
        return

    # --- Check Outputs ---
    print(f"\n--- ONNX Model Output Verification ---")
    print(f"Number of outputs received: {len(outputs)}")

    if len(outputs) == len(output_names):
        output_dict = {name: data for name, data in zip(output_names, outputs)}
        
        # Check for Detection Output
        det_output_name = 'output_detections' # Name used during export
        if det_output_name in output_dict:
            print(f"  Found '{det_output_name}'")
            print(f"    Shape: {output_dict[det_output_name].shape}")
            print(f"    Data type: {output_dict[det_output_name].dtype}")
        else:
            print(f"  WARNING: Expected detection output '{det_output_name}' not found!")
            
        # Check for Horizon Output
        hor_output_name = 'output_horizon' # Name used during export
        if hor_output_name in output_dict:
            print(f"  Found '{hor_output_name}'")
            print(f"    Shape: {output_dict[hor_output_name].shape}")
            print(f"    Data type: {output_dict[hor_output_name].dtype}")
            # Print a sample value
            if output_dict[hor_output_name].size > 0:
                 print(f"    Sample value (batch 0, item 0): {output_dict[hor_output_name][0, 0]}")
        else:
            print(f"  WARNING: Expected horizon output '{hor_output_name}' not found!")
            
    else:
        print(f"  ERROR: Number of outputs ({len(outputs)}) does not match expected number based on model graph ({len(output_names)}).")

    print("-------------------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='yolov7_horizon.onnx', help='Path to the ONNX model file')
    parser.add_argument('--image', type=str, default='inference/images/bus.jpg', help='Path to the input image file')
    parser.add_argument('--img-size', nargs=2, type=int, default=[640, 640], help='Image size (height width) for inference')
    
    opt = parser.parse_args()
    opt.model = Path(opt.model)
    opt.image = Path(opt.image)
    opt.img_size = tuple(opt.img_size)
    
    print(f"Running inference check with options: {opt}")
    main(opt) 