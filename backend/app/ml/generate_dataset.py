"""
Generate the full synthetic degraded dataset from clean images.

Reads data/splits/{train,val,test}.txt, applies every (issue_type, severity)
degradation from degrade.py to every clean image in each split, and writes:

  data/degraded/<split>/<issue_type>/sev<N>/<original_filename>
  data/labels.csv   -- manifest with one row per generated image

Also writes one "clean" entry per original image (severity=0, issue_type="none")
so the dataset includes acceptable examples, not just degraded ones.

Run from backend/:  python -m app.ml.generate_dataset
"""

import csv
import os
import sys

import cv2

# Allow running as a script from backend/ without package install
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.ml.degrade import generate_degraded_variants  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
CLEAN_DIR = os.path.join(DATA_DIR, "clean")
SPLITS_DIR = os.path.join(DATA_DIR, "splits")
DEGRADED_DIR = os.path.join(DATA_DIR, "degraded")
LABELS_CSV = os.path.join(DATA_DIR, "labels.csv")

RNG_SEED = 42


def read_split(split_name: str) -> list[str]:
    path = os.path.join(SPLITS_DIR, f"{split_name}.txt")
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def process_split(split_name: str, filenames: list[str], writer: csv.writer, seed_offset: int):
    for idx, filename in enumerate(filenames):
        src_path = os.path.join(CLEAN_DIR, filename)
        image = cv2.imread(src_path)
        if image is None:
            print(f"  WARNING: could not read {src_path}, skipping")
            continue

        # 1. Write the clean/acceptable example as-is
        clean_out_dir = os.path.join(DEGRADED_DIR, split_name, "none", "sev0")
        os.makedirs(clean_out_dir, exist_ok=True)
        clean_out_path = os.path.join(clean_out_dir, filename)
        cv2.imwrite(clean_out_path, image)
        writer.writerow([
            os.path.relpath(clean_out_path, DATA_DIR).replace("\\", "/"),
            split_name, filename, "none", 0, "acceptable",
        ])

        # 2. Write every degraded variant
        img_seed = RNG_SEED + seed_offset + idx
        for issue_type, severity, degraded_img in generate_degraded_variants(image, rng_seed=img_seed):
            out_dir = os.path.join(DEGRADED_DIR, split_name, issue_type, f"sev{severity}")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, filename)
            cv2.imwrite(out_path, degraded_img)

            # quality label heuristic from severity, for the overall label field
            if severity <= 2:
                quality_label = "degraded"
            else:
                quality_label = "potentially_defective"

            writer.writerow([
                os.path.relpath(out_path, DATA_DIR).replace("\\", "/"),
                split_name, filename, issue_type, severity, quality_label,
            ])

        if (idx + 1) % 100 == 0:
            print(f"  [{split_name}] processed {idx + 1}/{len(filenames)} clean images")


def main():
    os.makedirs(DEGRADED_DIR, exist_ok=True)
    splits = {name: read_split(name) for name in ("train", "val", "test")}
    for name, files in splits.items():
        print(f"{name}: {len(files)} clean images")

    with open(LABELS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "split", "source_filename", "issue_type", "severity", "quality_label"])

        seed_offset = 0
        for split_name, filenames in splits.items():
            print(f"Generating degraded variants for split: {split_name}")
            process_split(split_name, filenames, writer, seed_offset)
            seed_offset += len(filenames)

    print(f"\nDone. Labels manifest written to {LABELS_CSV}")
    total_expected = sum(len(f) for f in splits.values()) * (1 + 5 * 5)  # 1 clean + 5 issues x 5 severities
    print(f"Expected total rows in labels.csv: {total_expected}")


if __name__ == "__main__":
    main()