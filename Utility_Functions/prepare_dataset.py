# Creates the 90-degree-rotated NGB images referenced in train_filenames.txt
# name: prepare_dataset.py
# usage: python3 -m Utility_Functions.prepare_dataset   (from the repo root, after BMC-Dataset/ is flattened)

import os
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.path.join(ROOT, "BMC-Dataset")
SUFFIX = "_90Rotate.jpg"

def main():
    made = missing = 0
    for listfile in ("train_filenames.txt", "test_filenames.txt"):
        with open(os.path.join(ROOT, listfile)) as fd:
            names = [n.strip() for n in fd if n.strip()]
        for name in names:
            dst = os.path.join(DATASET, name)
            if os.path.exists(dst):
                continue
            if not name.endswith(SUFFIX):
                missing += 1
                print("missing original:", name)
                continue
            src = os.path.join(DATASET, name[:-len(SUFFIX)] + ".jpg")
            if not os.path.exists(src):
                missing += 1
                print("missing source for:", name)
                continue
            Image.open(src).rotate(90, expand=True).save(dst)
            made += 1
    print(f"created {made} rotated images, {missing} still missing")

if __name__ == "__main__":
    main()
