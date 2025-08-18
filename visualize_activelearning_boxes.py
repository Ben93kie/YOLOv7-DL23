import cv2
import os
import sys

# === Paths ===
# video_path = "/tmp/954_2_compressed.mp4"
# label_file = "/home/kiefer/Downloads/2024-05-20T16-12-20_category.txt"
video_path = "/mnt/gs/videos/56MekqrqWXUiOZhb3WPzIc0O78w2/dsf/2024-05-20T16-12-20/954_2_compressed.mp4"
label_file = "/mnt/gs/eventmode/56MekqrqWXUiOZhb3WPzIc0O78w2/dsf/2024-05-20T16-12-20/2024-05-20T16-12-20_category.txt"

output_dir = "/tmp/yolo_dataset2"

# === Check video path ===
if not os.path.isfile(video_path):
    sys.exit(f"❌ Error: Video file does not exist at path: {video_path}")

# === Prepare output structure ===
splits = ['train', 'val', 'test']
for split in splits:
    os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

# === Read labels ===
labels_by_frame = {}
with open(label_file, "r") as f:
    for line in f:
        parts = line.strip().split()
        frame_id = int(parts[0])
        category = parts[1]
        x_rel, y_rel, w_rel, h_rel = map(float, parts[2:])

        if frame_id not in labels_by_frame:
            labels_by_frame[frame_id] = []
        labels_by_frame[frame_id].append((category, x_rel, y_rel, w_rel, h_rel))

# === Category to ID ===
categories = sorted({label[0] for labels in labels_by_frame.values() for label in labels})
cat2id = {cat: i for i, cat in enumerate(categories)}
print("Detected classes:", cat2id)

# === Process video ===
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    sys.exit(f"❌ Error: Could not open video file: {video_path}")

frame_idx = 0
img_w, img_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx in labels_by_frame:
        # Draw boxes and show frame
        for category, x_rel, y_rel, w_rel, h_rel in labels_by_frame[frame_idx]:
            x1 = int(x_rel * img_w)
            y1 = int(y_rel * img_h)
            x2 = int((x_rel + w_rel) * img_w)
            y2 = int((y_rel + h_rel) * img_h)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, category, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        cv2.imshow("Annotated Frame", frame)
        if cv2.waitKey(1) == 27:  # ESC to quit
            break

        # Write to train/val/test
        for split in splits:
            img_dir = os.path.join(output_dir, 'images', split)
            lbl_dir = os.path.join(output_dir, 'labels', split)

            cv2.imwrite(os.path.join(img_dir, f"{frame_idx:06d}.jpg"), frame)
            with open(os.path.join(lbl_dir, f"{frame_idx:06d}.txt"), "w") as out_label:
                for category, x_rel, y_rel, w_rel, h_rel in labels_by_frame[frame_idx]:
                    cx_rel = x_rel + w_rel / 2
                    cy_rel = y_rel + h_rel / 2
                    out_label.write(f"{cat2id[category]} {cx_rel:.6f} {cy_rel:.6f} {w_rel:.6f} {h_rel:.6f}\n")

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()
print(f"✅ Done. YOLO dataset with train/val/test saved in: {output_dir}")