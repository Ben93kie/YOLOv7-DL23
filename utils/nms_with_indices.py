#!/usr/bin/env python3
"""
Enhanced NMS function that tracks indices for multi-head detection
"""

import time
import torch
import torchvision
from .general import xywh2xyxy, box_iou


def non_max_suppression_with_indices(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, 
                                   multi_label=False, labels=(), return_indices=True):
    """Runs Non-Maximum Suppression (NMS) on inference results with optional index tracking

    Args:
        prediction: Model predictions tensor [batch, anchors, predictions]
        conf_thres: Confidence threshold
        iou_thres: IoU threshold for NMS
        classes: Filter by class
        agnostic: Class-agnostic NMS
        multi_label: Multiple labels per box
        labels: Autolabelling
        return_indices: Whether to return original indices of surviving detections

    Returns:
        output: List of detections, on (n,6) tensor per image [xyxy, conf, cls]
        indices: List of original indices for each detection (if return_indices=True)
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    indices_output = [] if return_indices else None
    
    for xi, x in enumerate(prediction):  # image index, image inference
        # Track original indices before any filtering
        original_indices = torch.arange(x.shape[0], device=x.device)
        
        # Apply constraints
        x = x[xc[xi]]  # confidence filtering
        original_indices = original_indices[xc[xi]]  # track indices through confidence filtering

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)
            # Add dummy indices for autolabels (they don't correspond to original predictions)
            label_indices = torch.full((len(l),), -1, device=x.device)
            original_indices = torch.cat((original_indices, label_indices), 0)

        # If none remain process next image
        if not x.shape[0]:
            if return_indices:
                indices_output.append(torch.zeros((0,), dtype=torch.long, device=prediction.device))
            continue

        # Compute conf
        if nc == 1:
            x[:, 5:] = x[:, 4:5] # for models with one class, cls_loss is 0 and cls_conf is always 0.5,
                                 # so there is no need to multiplicate.
        else:
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
            original_indices = original_indices[i]  # track indices through multi-label filtering
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            conf_mask = conf.view(-1) > conf_thres
            x = torch.cat((box, conf, j.float()), 1)[conf_mask]
            original_indices = original_indices[conf_mask]  # track indices through confidence filtering

        # Filter by class
        if classes is not None:
            class_mask = (x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)
            x = x[class_mask]
            original_indices = original_indices[class_mask]  # track indices through class filtering

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            if return_indices:
                indices_output.append(torch.zeros((0,), dtype=torch.long, device=prediction.device))
            continue
        elif n > max_nms:  # excess boxes
            sort_indices = x[:, 4].argsort(descending=True)[:max_nms]
            x = x[sort_indices]  # sort by confidence
            original_indices = original_indices[sort_indices]  # track indices through sorting

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        nms_indices = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        
        if nms_indices.shape[0] > max_det:  # limit detections
            nms_indices = nms_indices[:max_det]
            
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[nms_indices], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[nms_indices, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                nms_indices = nms_indices[iou.sum(1) > 1]  # require redundancy

        # Final filtering - these are the detections that survive
        output[xi] = x[nms_indices]
        
        if return_indices:
            # These are the original indices of the surviving detections
            final_indices = original_indices[nms_indices]
            indices_output.append(final_indices)
        
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    if return_indices:
        return output, indices_output
    else:
        return output


