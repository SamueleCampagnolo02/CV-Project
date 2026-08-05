"""
Converte le annotazioni COCO di TACO in formato YOLO, mantenendo TUTTE
le categorie originali (nessun raggruppamento), e crea lo split train/val.

Le classi non vengono indovinate o scritte a mano: lo script legge
direttamente coco["categories"] dal file annotations.json, quindi usa
sempre gli ID e i nomi esatti presenti nel tuo file.
"""

import argparse
import json
import random
import shutil
from pathlib import Path


def build_category_map(coco):
    categories = sorted(coco["categories"], key=lambda c: c["id"])
    cat_id_to_yolo_idx = {}
    class_names_in_order = []
    for i, cat in enumerate(categories):
        cat_id_to_yolo_idx[cat["id"]] = i
        class_names_in_order.append(cat["name"])
    return cat_id_to_yolo_idx, class_names_in_order


def convert(taco_dir: Path, out_dir: Path, val_ratio: float, seed: int = 42):
    ann_path = taco_dir / "data" / "annotations.json"
    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    cat_map, class_names = build_category_map(coco)
    print(f"Trovate {len(class_names)} categorie nel file annotations.json:")
    for i, name in enumerate(class_names):
        print(f"  {i}: {name}")

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

    missing = 0
    written = 0

    for img_id in image_ids:
        img_info = images_by_id[img_id]
        split = "val" if img_id in val_ids else "train"

        src_path = taco_dir / "data" / img_info["file_name"]
        if not src_path.exists():
            missing += 1
            continue

        dst_name = f"{img_id}_{src_path.name}"
        dst_img_path = out_dir / "images" / split / dst_name
        shutil.copy2(src_path, dst_img_path)

        w, h = img_info["width"], img_info["height"]
        lines = []
        for ann in anns_by_image[img_id]:
            yolo_idx = cat_map[ann["category_id"]]
            x, y, bw, bh = ann["bbox"]
            x_center = (x + bw / 2) / w
            y_center = (y + bh / 2) / h
            norm_w = bw / w
            norm_h = bh / h
            lines.append(f"{yolo_idx} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

        label_path = out_dir / "labels" / split / (Path(dst_name).stem + ".txt")
        label_path.write_text("\n".join(lines), encoding="utf-8")
        written += 1

    print(f"Immagini convertite: {written}")
    print(f"Immagini mancanti su disco: {missing}")

    yaml_content = (
        f"path: {out_dir.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n"
        + "\n".join(f"  {i}: {name}" for i, name in enumerate(class_names))
        + "\n"
    )
    (out_dir / "data.yaml").write_text(yaml_content, encoding="utf-8")
    print(f"data.yaml creato in {out_dir / 'data.yaml'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--taco_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    args = parser.parse_args()

    convert(args.taco_dir, args.out_dir, args.val_ratio)
