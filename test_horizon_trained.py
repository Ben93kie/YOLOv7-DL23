import argparse
import time
from pathlib import Path
import os
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords, \
    xyxy2xywh, strip_optimizer, set_logging, increment_path, check_file
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized

def load_horizon_label(img_path):
    """Loads horizon label (y-intercept, slope) from txt file corresponding to img_path."""
    try:
        label_path = Path(img_path.replace('/images/', '/labels/').replace('\\images\\', '\\labels\\'))
        label_path = label_path.with_suffix('.txt')

        if not label_path.exists():
            print(f"Warning: Label file not found at {label_path}")
            return None

        with open(label_path, 'r') as f:
            line = f.readline().strip()
            # Handle both comma and space delimiters
            parts = line.replace(',', ' ').split() 
            if len(parts) == 2:
                return torch.tensor([float(parts[0]), float(parts[1])], dtype=torch.float32)
            else:
                print(f"Warning: Invalid format in label file {label_path}. Expected 2 floats, got: '{line}'")
                return None
    except Exception as e:
        print(f"Error loading horizon label for {img_path}: {e}")
        return None

def detect(opt):
    source, weights, view_img, save_txt, imgsz, cfg = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.cfg
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors (mainly for potential bbox plotting)
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():
            pred_det, pred_hor = model(img, augment=opt.augment) # Dual output
        t2 = time_synchronized()

        # Process horizon prediction
        horizon_pred = pred_hor.squeeze().cpu() # Shape [2]

        # Load ground truth horizon label
        horizon_gt_loaded = load_horizon_label(path)
        
        # --- DEBUGGING: Use dummy GT if real label not found/invalid ---
        if horizon_gt_loaded is None:
            print(f"  INFO: Real horizon label not found/invalid for {path}. Using dummy target [0.5, 0.1] for comparison.")
            horizon_gt = torch.tensor([0.5, 0.1], dtype=torch.float32)
        else:
            horizon_gt = horizon_gt_loaded
        # --- END DEBUGGING ---

        print(f"--- Image: {path} ---")
        # Display the actual loaded GT if available
        if horizon_gt_loaded is not None:
            print(f"  Ground Truth Horizon (Loaded): y_intercept={horizon_gt_loaded[0]:.4f}, slope={horizon_gt_loaded[1]:.4f}")
        else:
            print(f"  Ground Truth Horizon (Loaded): Not found or error loading.")
        
        # Print the prediction and the GT being used for error calculation
        print(f"  Predicted Horizon:           y_intercept={horizon_pred[0]:.4f}, slope={horizon_pred[1]:.4f}")
        print(f"  Ground Truth (for Error):    y_intercept={horizon_gt[0]:.4f}, slope={horizon_gt[1]:.4f}")

        # Always calculate error against horizon_gt (which is either real or dummy)
        mse = torch.mean((horizon_pred - horizon_gt)**2)
        mae = torch.mean(torch.abs(horizon_pred - horizon_gt))
        print(f"  Error vs GT for Error: MSE={mse:.6f}, MAE={mae:.6f}") # Increased precision for error
        print(f"  Inference Time: {(t2 - t1):.3f}s")
        print("---------------------")

        # --- Optional: Detection processing and saving ---
        # Apply NMS to detections
        pred_det = non_max_suppression(pred_det[0] if isinstance(pred_det, tuple) else pred_det, 
                                       opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred_det):  # detections per image
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path
            save_path = str(Path(opt.project) / opt.name / p.name)  # img.jpg
            txt_path = str(Path(opt.project) / opt.name / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results (optional)
                if save_img or save_txt or view_img:
                    Path(Path(opt.project) / opt.name / 'labels').mkdir(parents=True, exist_ok=True) # Create labels dir
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or view_img:  # Add bbox to image
                            label = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
            
            # Save image with predictions (optional)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print(f'Results saved to {Path(opt.project) / opt.name}')

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--project', default='runs/test_horizon', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--cfg', type=str, default='', help='(optional) model.yaml path, overrides model.yaml in weights') # Add cfg option
    opt = parser.parse_args()
    print(opt)
    # check_requirements(exclude=('pycocotools', 'thop')) # Might be needed if running standalone

    # Add default cfg if not provided and needed for model loading without ckpt yaml
    if not opt.cfg and not Path(opt.weights[0]).suffix == '.pt':
         # Try to infer from weights name or use a default? For now, require it if no ckpt yaml.
         print("Warning: --cfg not provided. Attempting to load model structure from weights. If this fails, please provide the corresponding --cfg file.")
         # Example default if needed: opt.cfg = 'cfg/training/yolov7-horizon-deploybased.yaml'

    with torch.no_grad():
        detect(opt=opt) 