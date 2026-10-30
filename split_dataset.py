"""
split_dataset.py

Splits raw category folders (dataset/<class>/*.jpg) into
dataset_split/train, dataset_split/val, dataset_split/test
using a 70/15/15 split.

Usage:
    python split_dataset.py

Expected input structure:
    dataset/
        apple/
        banana/
        bread/
        egg/
        milk_carton/
        tomato/
        packaged_snack/
        canned_food/
"""

import os
import shutil
import random

SOURCE_DIR = "dataset"
OUTPUT_DIR = "dataset_split"
SPLIT_RATIOS = (0.70, 0.15, 0.15)  # train, val, test
SEED = 42

random.seed(SEED)


def split_dataset():
    if not os.path.isdir(SOURCE_DIR):
        raise FileNotFoundError(
            f"Expected a '{SOURCE_DIR}' folder with one subfolder per class."
        )

    classes = [
        d for d in os.listdir(SOURCE_DIR)
        if os.path.isdir(os.path.join(SOURCE_DIR, d))
    ]

    if not classes:
        raise ValueError(f"No class subfolders found inside '{SOURCE_DIR}'.")

    print(f"Found {len(classes)} classes: {classes}")

    total_images = sum(
        len([f for f in os.listdir(os.path.join(SOURCE_DIR, c))
             if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        for c in classes
    )
    if total_images == 0:
        raise SystemExit(
            "\nNo images found yet! Put your photos (.jpg/.jpeg/.png) inside the "
            "folders in 'dataset/' (e.g. dataset/apple/), then run this again."
        )

    for split in ("train", "val", "test"):
        for cls in classes:
            os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)

    summary = {}

    for cls in classes:
        class_dir = os.path.join(SOURCE_DIR, cls)
        images = [
            f for f in os.listdir(class_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        random.shuffle(images)

        n = len(images)
        n_train = int(n * SPLIT_RATIOS[0])
        n_val = int(n * SPLIT_RATIOS[1])

        train_files = images[:n_train]
        val_files = images[n_train:n_train + n_val]
        test_files = images[n_train + n_val:]

        for fname in train_files:
            shutil.copy(
                os.path.join(class_dir, fname),
                os.path.join(OUTPUT_DIR, "train", cls, fname),
            )
        for fname in val_files:
            shutil.copy(
                os.path.join(class_dir, fname),
                os.path.join(OUTPUT_DIR, "val", cls, fname),
            )
        for fname in test_files:
            shutil.copy(
                os.path.join(class_dir, fname),
                os.path.join(OUTPUT_DIR, "test", cls, fname),
            )

        summary[cls] = {
            "total": n,
            "train": len(train_files),
            "val": len(val_files),
            "test": len(test_files),
        }

    print("\nSplit summary:")
    for cls, counts in summary.items():
        print(f"  {cls}: {counts}")

    print(f"\nDone. Split data written to '{OUTPUT_DIR}/'.")


if __name__ == "__main__":
    split_dataset()
