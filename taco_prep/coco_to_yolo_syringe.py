"""
Converts the custom syringe dataset (COCO JSON exported from MakeSense.ai)
into YOLO format, as a standalone single-class dataset kept separate from
TACO. Creates a train/val split and a data.yaml with a single class.

Usage:
    python coco_to_yolo_syringe.py --custom_dir custom_syringe --out_dir custom_syringe_yolo --val_ratio 0.2
"""

import argparse
import json
import random
import shutil
from pathlib import Path

CLASS_NAMES = ["syringe"]
def convert(custom_dir: Path, out_dir: Path, val_ratio: float, seed: int = 42):
    ann_path = custom_dir / "annotations.json"
    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images_by_id = {img["id"]: img for img in coco["images"]}
    anns_by_image = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    image_ids = [img_id for img_id in images_by_id if img_id in anns_by_image]
    random.seed(seed)
    random.shuffle(image_ids)

    n_val = int(len(image_ids) * val_ratio)
    val_ids = set(image_ids[:n_val])

    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    copied = 0
    missing = 0

    for img_id in image_ids:
        img_info = images_by_id[img_id]
        split = "val" if img_id in val_ids else "train"

        src_path = custom_dir / "images" / img_info["file_name"]
        if not src_path.exists():
            missing += 1
            continue

        dst_img = out_dir / "images" / split / src_path.name
        shutil.copy2(src_path, dst_img)

        w, h = img_info["width"], img_info["height"]
        lines = []
        for ann in anns_by_image[img_id]:
            x, y, bw, bh = ann["bbox"]  # COCO: x_min, y_min, width, height
            x_center = (x + bw / 2) / w
            y_center = (y + bh / 2) / h
            norm_w = bw / w
            norm_h = bh / h
            # single class dataset -> class index is always 0
            lines.append(f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

        dst_label = out_dir / "labels" / split / (src_path.stem + ".txt")
        dst_label.write_text("\n".join(lines), encoding="utf-8")

        copied += 1

    print(f"Images converted: {copied}")
    print(f"Images missing on disk: {missing}")

    yaml_content = (
        f"path: {out_dir.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n"
        + "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES))
        + "\n"
    )
    (out_dir / "data.yaml").write_text(yaml_content, encoding="utf-8")
    print(f"data.yaml created at {out_dir / 'data.yaml'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--custom_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--ann_path", type=Path, default=None,
                         help="Path to the annotations JSON file (default: <custom_dir>/annotations.json)")
    args = parser.parse_args()

    convert(args.custom_dir, args.out_dir, args.val_ratio, args.ann_path)