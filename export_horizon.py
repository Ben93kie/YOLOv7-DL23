"""Exports a YOLOv7+Horizon model architecture to ONNX format.

Based on the standard YOLOv7 export.py, but adapted for the dual-output
(detection + horizon) model and focuses on exporting the architecture
defined in a YAML config file, without loading trained weights.
"""

import argparse
import sys
import os
from pathlib import Path

import torch
import torch.nn as nn

# Add yolov7 root directory to path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # Assuming script is in yolov7 root
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative path

# Import Model class after adding ROOT to path
from models.yolo import Model
from utils.torch_utils import select_device
# from utils.general import check_requirements # Optional: If checking onnx, onnxsim

# --- ONNX Export Function ---
def export_onnx(model, im, file, opset, dynamic, simplify):
    # Standard ONNX export
    try:
        print(f'\nStarting ONNX export with opset {opset}...')
        f = str(file)  # Ensure Path is string

        # Define input and output names
        input_names = ['images']
        output_names = ['output_detections', 'output_horizon']

        # Define dynamic axes if requested
        dynamic_axes = None
        if dynamic:
            dynamic_axes = {
                'images': {0: 'batch'}, # Batch axis for input
                'output_detections': {0: 'batch'}, # Batch axis for detection output
                'output_horizon': {0: 'batch'} # Batch axis for horizon output
            }
            print(f'    Using dynamic axes: {dynamic_axes}')

        torch.onnx.export(
            model,
            im,
            f,
            verbose=False,
            opset_version=opset,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes
        )

        # Checks
        import onnx
        model_onnx = onnx.load(f)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model
        print(f'    ONNX model checked OK.')

        # Simplify
        if simplify:
            try:
                import onnxsim
                print(f'    Simplifying ONNX model with onnx-simplifier...')
                model_onnx, check = onnxsim.simplify(model_onnx)
                assert check, 'assert check failed'
                onnx.save(model_onnx, f)
                print(f'    ONNX simplification successful.')
            except Exception as e:
                print(f'    ONNX simplification failed: {e}')

        print(f'ONNX export success, saved as {f}')
        return f # Return final path

    except Exception as e:
        print(f'ONNX export failed: {e}')
        return None

# --- Main Function ---
def main(opt):
    # Setup
    device = select_device(opt.device)
    output_path = Path(opt.f)
    output_path.parent.mkdir(parents=True, exist_ok=True) # Create output dir if needed

    # Load model architecture from config
    print(f'Loading model structure from {opt.cfg}...')
    model = Model(opt.cfg, ch=3).to(device) # Assumes nc/anchors from YAML config
    print(f'Model loaded successfully.')

    # Configure model for export
    model.eval()
    for k, m in model.named_modules(): # Set trace=False for modules (may not be strictly needed for all)
         m.traced = False
         if isinstance(m, (Model)):
             m.onnx_dynamic = opt.dynamic
             # m.export = True # May or may not be needed depending on head implementation

    # --- Dummy Input Tensor ---
    img_size = opt.img_size * 2 if len(opt.img_size) == 1 else opt.img_size  # expand if single value
    im = torch.zeros(opt.batch_size, 3, *img_size).to(device)  # BCHW
    print(f'Creating dummy input tensor with shape: {im.shape}')

    # --- Perform Export ---
    exported_file = export_onnx(model, im, output_path, opt.opset, opt.dynamic, opt.simplify)

    if exported_file:
        print(f'\n✅ Export complete. Model saved to: {exported_file}')
    else:
        print(f'\n❌ Export failed.')


# --- Argument Parser ---
def parse_opt():
    parser = argparse.ArgumentParser()
    # Default config path points to the horizon model config
    parser.add_argument('--cfg', type=str, default=ROOT / 'cfg/training/yolov7-horizon-deploybased.yaml', help='model.yaml path defining architecture')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image (h, w)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size for dummy input')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--f', type=str, default=ROOT / 'yolov7_horizon.onnx', help='output ONNX file path')
    parser.add_argument('--opset', type=int, default=12, help='ONNX opset version')
    parser.add_argument('--dynamic', action='store_true', help='export with ONNX dynamic axes')
    parser.add_argument('--simplify', action='store_true', help='simplify ONNX model using onnxsim')
    # Note: Removed --weights argument as this script focuses on exporting architecture from YAML
    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1 # expand single image size argument
    opt.f = str(Path(opt.f).with_suffix('.onnx')) # Ensure .onnx suffix
    print(f"Export options: {opt}")
    return opt

# --- Entry Point ---
if __name__ == "__main__":
    # check_requirements(('onnx', 'onnx-simplifier')) # Optional check
    opt = parse_opt()
    main(opt) 