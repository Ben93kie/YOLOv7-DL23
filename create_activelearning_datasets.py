#!/usr/bin/env python3
import os
import cv2
import sys
from pathlib import Path

def process_label_file(label_file: Path, video_file: Path, output_dir: Path):
    """
    Reads one detection file and its corresponding video,
    creates a YOLO-format dataset (train/val/test splits duplicated).
    """
    # Parse labels
    labels_by_frame = {}
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            frame_id = int(parts[0])
            category = parts[1]
            x_rel, y_rel, w_rel, h_rel = map(float, parts[2:])
            labels_by_frame.setdefault(frame_id, []).append((category, x_rel, y_rel, w_rel, h_rel))

    # Map categories to integer IDs
    categories = sorted({cat for labels in labels_by_frame.values() for cat, *_ in labels})
    cat2id = {cat: i for i, cat in enumerate(categories)}
    print(f"Processing {label_file.name}: classes {cat2id}")

    # Prepare output directories
    splits = ['train', 'val', 'test']
    for split in splits:
        (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video {video_file}", file=sys.stderr)
        return

    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_idx = 0

    # Process frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in labels_by_frame:
            # Draw boxes for sanity check
            for category, x_rel, y_rel, w_rel, h_rel in labels_by_frame[frame_idx]:
                x1 = int(x_rel * img_w)
                y1 = int(y_rel * img_h)
                x2 = int((x_rel + w_rel) * img_w)
                y2 = int((y_rel + h_rel) * img_h)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, category, (x1, max(0, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            # Save images and labels to each split
            for split in splits:
                img_out = output_dir / 'images' / split / f"{frame_idx:06d}.jpg"
                lbl_out = output_dir / 'labels' / split / f"{frame_idx:06d}.txt"

                cv2.imwrite(str(img_out), frame)
                with open(lbl_out, 'w') as out_f:
                    for category, x_rel, y_rel, w_rel, h_rel in labels_by_frame[frame_idx]:
                        cx_rel = x_rel + w_rel / 2
                        cy_rel = y_rel + h_rel / 2
                        out_f.write(f"{cat2id[category]} {cx_rel:.6f} {cy_rel:.6f} "
                                    f"{w_rel:.6f} {h_rel:.6f}\n")

        frame_idx += 1

    cap.release()


def main():
    # Base directories (adjust these to your mount point/project)
    eventmode_base = Path("/mnt/gs/eventmode/56MekqrqWXUiOZhb3WPzIc0O78w2/dsf")
    videos_base   = Path("/mnt/gs/videos/56MekqrqWXUiOZhb3WPzIc0O78w2/dsf")
    output_root   = Path("/tmp/yolo_datasets")

    for event_dir in sorted(eventmode_base.iterdir()):
        if not event_dir.is_dir():
            continue

        # Compute matching video folder
        rel_path = event_dir.relative_to(eventmode_base)
        video_dir = videos_base / rel_path

        # Find the single video in that folder
        video_files = list(video_dir.glob("*_compressed.mp4"))
        if not video_files:
            print(f"⚠️  Warning: No video found in {video_dir}", file=sys.stderr)
            continue
        video_file = video_files[0]

        # For each detection file in this event folder, build its own dataset
        for label_file in sorted(event_dir.glob("*.txt")):
            dataset_name = label_file.stem
            output_dir = output_root / dataset_name
            process_label_file(label_file, video_file, output_dir)

    print("✅ All datasets created under:", output_root)


if __name__ == "__main__":
    main()
