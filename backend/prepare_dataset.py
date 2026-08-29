import os
import random
import shutil


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
SAMPLE_SIZE = 1000

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))

COCO_DIR = os.path.join(
    BACKEND_DIR,
    "data",
    "coco_raw",
    "val2017"
)

CLEAN_DIR = os.path.join(
    BACKEND_DIR,
    "data",
    "clean"
)

SPLITS_DIR = os.path.join(
    BACKEND_DIR,
    "data",
    "splits"
)

TRAIN_FILE = os.path.join(SPLITS_DIR, "train.txt")
VAL_FILE = os.path.join(SPLITS_DIR, "val.txt")
TEST_FILE = os.path.join(SPLITS_DIR, "test.txt")


# ============================================================
# SETUP
# ============================================================

random.seed(SEED)

os.makedirs(CLEAN_DIR, exist_ok=True)
os.makedirs(SPLITS_DIR, exist_ok=True)


# ============================================================
# FIND COCO IMAGES
# ============================================================

if not os.path.isdir(COCO_DIR):
    raise FileNotFoundError(
        f"COCO directory not found:\n{COCO_DIR}\n\n"
        "Make sure val2017.zip has been extracted correctly."
    )

all_images = []

for filename in os.listdir(COCO_DIR):
    if filename.lower().endswith(".jpg"):
        all_images.append(filename)

all_images.sort()

print(f"Found {len(all_images)} COCO images.")


# ============================================================
# RANDOMLY SAMPLE 1000 IMAGES
# ============================================================

if len(all_images) < SAMPLE_SIZE:
    raise RuntimeError(
        f"Only found {len(all_images)} images, "
        f"but need {SAMPLE_SIZE}."
    )

selected = random.sample(all_images, SAMPLE_SIZE)

# Sort after sampling so the resulting clean directory
# and split files are easy to inspect.
selected.sort()

print(f"Selected {len(selected)} images using seed {SEED}.")


# ============================================================
# COPY SELECTED IMAGES INTO clean/
# ============================================================

for filename in selected:
    source = os.path.join(COCO_DIR, filename)
    destination = os.path.join(CLEAN_DIR, filename)

    shutil.copy2(source, destination)

print(f"Copied {len(selected)} images into:")
print(CLEAN_DIR)


# ============================================================
# CREATE 70 / 15 / 15 SPLIT
# ============================================================

# Shuffle a copy so the split itself is randomized.
split_images = selected[:]
random.shuffle(split_images)

train_count = int(len(split_images) * 0.70)
val_count = int(len(split_images) * 0.15)

train_images = split_images[:train_count]
val_images = split_images[train_count:train_count + val_count]
test_images = split_images[train_count + val_count:]


# ============================================================
# WRITE SPLIT FILES
# ============================================================

def write_split(filename, images):
    with open(filename, "w", encoding="utf-8") as f:
        for image in images:
            f.write(image + "\n")


write_split(TRAIN_FILE, train_images)
write_split(VAL_FILE, val_images)
write_split(TEST_FILE, test_images)


# ============================================================
# VERIFY SPLITS
# ============================================================

train_set = set(train_images)
val_set = set(val_images)
test_set = set(test_images)

overlap_train_val = train_set & val_set
overlap_train_test = train_set & test_set
overlap_val_test = val_set & test_set

print()
print("========== DATASET SUMMARY ==========")
print(f"Clean images : {len(selected)}")
print(f"Train        : {len(train_set)}")
print(f"Validation   : {len(val_set)}")
print(f"Test         : {len(test_set)}")
print()

print("========== SPLIT VERIFICATION ==========")

if not overlap_train_val:
    print("Train vs Val   : NO OVERLAP")
else:
    print("Train vs Val   : OVERLAP FOUND")

if not overlap_train_test:
    print("Train vs Test  : NO OVERLAP")
else:
    print("Train vs Test  : OVERLAP FOUND")

if not overlap_val_test:
    print("Val vs Test    : NO OVERLAP")
else:
    print("Val vs Test    : OVERLAP FOUND")

all_unique = (
    len(train_set | val_set | test_set)
    == len(selected)
)

if all_unique:
    print()
    print("SUCCESS: No filename appears in more than one split.")
else:
    print()
    print("ERROR: Duplicate filename detected across splits.")


# ============================================================
# FINAL PATHS
# ============================================================

print()
print("========== FILES CREATED ==========")
print(TRAIN_FILE)
print(VAL_FILE)
print(TEST_FILE)