import os
import argparse
from pathlib import Path

def generate_dummy_labels(img_dir, label_out_dir):
    """
    Generates dummy horizon label files for images in a directory.

    Args:
        img_dir (str): Path to the directory containing image files.
        label_out_dir (str): Path to the directory where dummy label files will be created.
    """
    img_dir_path = Path(img_dir)
    label_out_dir_path = Path(label_out_dir)

    if not img_dir_path.is_dir():
        print(f"Error: Image directory not found: {img_dir}")
        return

    # Create the output directory if it doesn't exist
    label_out_dir_path.mkdir(parents=True, exist_ok=True)
    print(f"Ensured output directory exists: {label_out_dir_path}")

    count = 0
    img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng', '.webp', '.mpo'] # From datasets.py

    for img_file in img_dir_path.iterdir():
        if img_file.is_file() and img_file.suffix.lower() in img_formats:
            # Construct the corresponding label file path
            label_file_name = img_file.stem + ".txt"
            label_file_path = label_out_dir_path / label_file_name

            # Write the dummy label
            try:
                with open(label_file_path, 'w') as f:
                    f.write("0.0,0.0")
                count += 1
            except Exception as e:
                print(f"Error writing label file {label_file_path}: {e}")

    print(f"Successfully generated {count} dummy label files in {label_out_dir_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dummy horizon labels for YOLOv7 training.")
    parser.add_argument("--img-dir", type=str, required=True,
                        help="Path to the directory containing training images.")
    parser.add_argument("--label-out-dir", type=str, required=True,
                        help="Path to the directory where dummy horizon label files will be saved.")

    args = parser.parse_args()

    generate_dummy_labels(args.img_dir, args.label_out_dir) 