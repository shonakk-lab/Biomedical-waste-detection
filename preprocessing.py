# ==========================================================
# Biomedical Waste Image Preprocessing Pipeline (Section 2.3)
# ==========================================================

import os
import cv2
import numpy as np
import hashlib
import random
import xml.etree.ElementTree as ET
from glob import glob
from sklearn.model_selection import train_test_split
from skimage.metrics import structural_similarity as ssim

# -------------------------------
# 1. Reproducibility (Seed Control)
# -------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# -------------------------------
# 2. Paths (Modify accordingly)
# -------------------------------
DATA_SOURCES = ["data/source1", "data/source2"]
OUTPUT_DIR = "data/processed"
ANNOTATION_DIR = "data/annotations"

IMG_SIZE = (224, 224)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------
# 3. Data Collection & Aggregation
# -------------------------------
def collect_images():
    image_paths = []
    for src in DATA_SOURCES:
        image_paths.extend(glob(os.path.join(src, "*.jpg")))
    print("Total collected images:", len(image_paths))
    return image_paths

# -------------------------------
# 4. Duplicate Removal (Hash)
# -------------------------------
def image_hash(img):
    return hashlib.md5(img.tobytes()).hexdigest()

def remove_duplicates(paths):
    unique = {}
    filtered = []

    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        h = image_hash(img)
        if h not in unique:
            unique[h] = p
            filtered.append(p)

    print("After duplicate removal:", len(filtered))
    return filtered

# -------------------------------
# 5. Near-Duplicate Removal (SSIM)
# -------------------------------
def remove_near_duplicates(paths, threshold=0.95):
    filtered = []

    for p in paths:
        img1 = cv2.imread(p, 0)
        img1 = cv2.resize(img1, (128, 128))

        duplicate = False
        for f in filtered:
            img2 = cv2.imread(f, 0)
            img2 = cv2.resize(img2, (128, 128))

            score, _ = ssim(img1, img2, full=True)
            if score > threshold:
                duplicate = True
                break

        if not duplicate:
            filtered.append(p)

    print("After SSIM filtering:", len(filtered))
    return filtered

# -------------------------------
# 6. Image Enhancement
# -------------------------------
def enhance_image(img):
    # Gaussian Blur (Eq. 3)
    img_blur = cv2.GaussianBlur(img, (5, 5), sigmaX=1)

    # Contrast Adjustment (Eq. 4)
    alpha = 1.3  # contrast factor
    beta = 15    # brightness offset
    img_contrast = cv2.convertScaleAbs(img_blur, alpha=alpha, beta=beta)

    return img_contrast

# -------------------------------
# 7. Resize & Normalize
# -------------------------------
def preprocess_image(img):
    # Resize (Eq. 5)
    img_resized = cv2.resize(img, IMG_SIZE)

    # Normalize (Eq. 6)
    img_norm = img_resized / 255.0

    return img_norm

# -------------------------------
# 8. Data Augmentation (Training Only)
# -------------------------------
def augment_image(img):
    augmented = []

    h, w = img.shape[:2]

    # Rotation (Eq. 7)
    angle = 20
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    rotated = cv2.warpAffine(img, M, (w, h))
    augmented.append(rotated)

    # Flipping (Eq. 8)
    flipped = cv2.flip(img, 1)
    augmented.append(flipped)

    return augmented

# -------------------------------
# 9. Annotation Parsing (LabelImg XML)
# -------------------------------
def parse_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    boxes = []
    labels = []

    for obj in root.findall("object"):
        label = obj.find("name").text

        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label)

    return boxes, labels

# -------------------------------
# 10. Bounding Box Validation (Eq. 9–12)
# -------------------------------
def validate_boxes(boxes, img_shape):
    H, W = img_shape[:2]
    valid_boxes = []

    for (xmin, ymin, xmax, ymax) in boxes:
        xmin = max(0, xmin)
        xmax = min(W, xmax)
        ymin = max(0, ymin)
        ymax = min(H, ymax)

        valid_boxes.append([xmin, ymin, xmax, ymax])

    return valid_boxes

# -------------------------------
# 11. Main Processing Pipeline
# -------------------------------
def process_dataset():
    paths = collect_images()
    paths = remove_duplicates(paths)
    paths = remove_near_duplicates(paths)

    data = []
    labels = []

    for path in paths:
        img = cv2.imread(path)

        # Enhancement
        img = enhance_image(img)

        # Preprocessing
        img = preprocess_image(img)

        data.append(img)
        labels.append(0)  # Modify based on class

    return np.array(data), np.array(labels)

# -------------------------------
# 12. Dataset Splitting (70-15-15)
# -------------------------------
def split_data(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=SEED
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=SEED
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

# -------------------------------
# 13. Apply Augmentation ONLY to Training Set
# -------------------------------
def augment_training(X_train, y_train):
    X_aug, y_aug = [], []

    for img, label in zip(X_train, y_train):
        aug_imgs = augment_image((img * 255).astype(np.uint8))
        for a in aug_imgs:
            a = preprocess_image(a)
            X_aug.append(a)
            y_aug.append(label)

    return np.concatenate([X_train, np.array(X_aug)]), np.concatenate([y_train, np.array(y_aug)])

# -------------------------------
# 14. Save Processed Data
# -------------------------------
def save_data(X, y, name):
    np.save(os.path.join(OUTPUT_DIR, f"{name}_X.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, f"{name}_y.npy"), y)

# -------------------------------
# 15. Run Full Pipeline
# -------------------------------
if __name__ == "__main__":
    X, y = process_dataset()

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Apply augmentation ONLY on training set (leakage prevention)
    X_train, y_train = augment_training(X_train, y_train)

    save_data(X_train, y_train, "train")
    save_data(X_val, y_val, "val")
    save_data(X_test, y_test, "test")

    print("Biomedical Waste Preprocessing Completed Successfully!")