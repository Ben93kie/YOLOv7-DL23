import argparse
import torch
import cv2
import numpy as np
from models.yolo import Model
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box

def test_horizon(opt):
    # Load model
    device = torch.device(f'cuda:{opt.device}' if torch.cuda.is_available() and opt.device != 'cpu' else 'cpu')
    print(f"Using device: {device}")
    
    # Use the deploy-based config
    # Note: Using a hardcoded config path here. Consider making it an arg if needed.
    model = Model('cfg/training/yolov7-horizon-deploybased.yaml', ch=3).to(device)  
    
    # --- Skip loading pre-trained weights for architecture test ---
    # ckpt = torch.load('yolov7.pt', map_location=device)
    # # Load matching weights, ignore missing keys for new layers
    # model.load_state_dict(ckpt['model'].float().state_dict(), strict=False) 
    model.eval() # Set to evaluation mode
    
    # Load test image
    # Note: Using a hardcoded image path. Consider making it an arg.
    img_path = 'inference/images/bus.jpg' 
    img0 = cv2.imread(img_path)  
    if img0 is None:
        print(f"Error: Could not load test image: {img_path}")
        return
    print(f"Loaded test image: {img_path}")
    
    # Preprocess single image
    img_size = (opt.img, opt.img) # Assuming square image size
    img = cv2.resize(img0, img_size)
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0) # Add batch dim: [1, 3, H, W]
        
    # Create batch by repeating the single image
    if opt.batch > 1:
        print(f"Repeating image to create batch size: {opt.batch}")
        batch_tensor = img.repeat_interleave(opt.batch, dim=0)
    else:
        batch_tensor = img
        
    print(f"Input batch tensor shape: {batch_tensor.shape}")
    
    # Inference
    with torch.no_grad():
        print("Running inference...")
        # Explicitly disable augmentation for testing
        pred = model(batch_tensor, augment=False)
        print("Inference complete.")
    
    # --- Print Output Shapes ---
    print(f"\n--- Model Output Information ---")
    print(f"Type of pred: {type(pred)}")
    if isinstance(pred, tuple) and len(pred) == 2:
        detections, horizon = pred
        print(f"  Detections tensor type: {type(detections)}")
        print(f"  Detections tensor shape: {detections.shape}")
        print(f"  Horizon tensor type: {type(horizon)}")
        print(f"  Horizon tensor shape: {horizon.shape}")
        
        # Basic check on horizon output values (first item in batch, first pixel)
        print(f"  Sample Horizon output (batch 0, pixel 0): {horizon[0, 0].cpu().numpy()}")
        
    else:
        print(f"  Unexpected output format. pred value: {pred}")
    print(f"-------------------------------\n")
    # --- End Print Output Shapes ---

    # --- Output Processing & Visualization Removed for Batch Test ---
    # This section assumes batch size 1 and involves NMS/drawing,
    # skipping it to focus on checking batch forward pass.
    
    print("Batch test script finished successfully.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=1, help='batch size')
    parser.add_argument('--img', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # Add other arguments from the original command if needed, e.g.:
    # parser.add_argument('--data', type=str, default='data/horizon.yaml', help='(optional) dataset.yaml path')
    # parser.add_argument('--weights', type=str, default='yolov7_horizon.pt', help='(optional) model.pt path')
    # parser.add_argument('--conf', type=float, default=0.001, help='(optional) confidence threshold')
    # parser.add_argument('--iou', type=float, default=0.65, help='(optional) NMS IoU threshold')
    # parser.add_argument('--name', default='yolov7_horizon_testing', help='(optional) save to runs/test/exp<n>/...')
    # parser.add_argument('--task', default='test', help='(optional) test, val, study, ...')
    
    opt = parser.parse_args()
    print(f"Running test with options: {opt}")
    
    test_horizon(opt) 