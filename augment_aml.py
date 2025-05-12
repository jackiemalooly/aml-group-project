# YOLO Data Augmentation and Balancing Pipeline
# ---------------------------------------------------
# Task: Take a YOLO-format dataset and balance the training set
# by augmenting underrepresented classes using Albumentations.
# The script outputs a new dataset folder with:
# - Original training images and labels
# - Augmented images/labels to balance all classes
# - Visualizations before and after augmentation

import os
import cv2
import yaml
import random
import shutil
import albumentations as A
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict

# ==================== Configuration ====================
DATA_YAML = '/user/HS402/rs02294/AML/aml-group-project/datasets/data.yaml'  # <-- CHANGE THIS
OUTPUT_DIR = 'datasets/augmented_dataset'
VISUALIZE_COUNT = 1

# Create output folders
os.makedirs(os.path.join(OUTPUT_DIR, 'train/images'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'train/labels'), exist_ok=True)

# ==================== Utils ====================
# def yolo_to_voc(box, w, h):
#     x, y, bw, bh = box
#     x_min = (x - bw / 2) * w
#     y_min = (y - bh / 2) * h
#     x_max = (x + bw / 2) * w
#     y_max = (y + bh / 2) * h
#     return [x_min, y_min, x_max, y_max]

# def voc_to_yolo(box, w, h):
#     x_min, y_min, x_max, y_max = box
#     x = ((x_min + x_max) / 2) / w
#     y = ((y_min + y_max) / 2) / h
#     bw = (x_max - x_min) / w
#     bh = (y_max - y_min) / h
#     return [x, y, bw, bh]

# ==================== Visualizations ====================
BOX_COLOR = (255, 0, 0)
TEXT_COLOR = (255, 255, 255)

def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    ((tw, th), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * th)), (x_min + tw, y_min), BOX_COLOR, -1)
    cv2.putText(img, text=class_name, org=(x_min, y_min - int(0.3 * th)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.35,
                color=TEXT_COLOR, lineType=cv2.LINE_AA)
    return img

def visualize(image, bboxes, category_ids, category_id_to_name):
    img = image.copy()
    for bbox, cls_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[cls_id]
        img = visualize_bbox(img, bbox, class_name)
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

def summarize_distribution(label_dir, class_names):
    counts = defaultdict(int)
    for f in os.listdir(label_dir):
        if f.endswith('.txt'):
            with open(os.path.join(label_dir, f)) as file:
                for line in file:
                    cls = int(float(line.strip().split()[0]))  
                    counts[cls] += 1
    print("\nClass Distribution:")
    for i in range(len(class_names)):
        print(f"Class {i:02d} ({class_names[i]}): {counts[i]}")
    return counts


# ==================== Albumentations Augmentations ====================
def get_augmentation_pipeline():
    return A.Compose([
        A.RandomSizedBBoxSafeCrop(height=640, width=640, erosion_rate=0.2, p=0.3),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomBrightnessContrast(p=0.4),
        A.Rotate(limit=15, p=0.3),
        A.Blur(blur_limit=3, p=0.2),
        A.ToGray(p=0.1),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5),
    ], bbox_params=A.BboxParams(
        format='yolo',  
        label_fields=['class_labels'],
        min_visibility=0.1,
        clip=True
    ))


# ==================== Main Pipeline ====================
with open(DATA_YAML, 'r') as f:
    data_cfg = yaml.safe_load(f)

train_img_dir = data_cfg['train']
train_lbl_dir = os.path.join(os.path.dirname(train_img_dir), 'labels')
class_names = data_cfg['names']
img_files = [f for f in os.listdir(train_img_dir) if f.lower().endswith(('.jpg', '.png'))]

class_counts = defaultdict(int)
data = []

# Load and copy original data
print("\nCopying original data...")
print("Image directory:", train_img_dir)
print("Label directory:", train_lbl_dir)
print("Number of images:", len(img_files))

for img_file in tqdm(img_files):
    img_path = os.path.join(train_img_dir, img_file)
    lbl_path = os.path.join(train_lbl_dir, img_file.replace('.jpg', '.txt').replace('.png', '.txt'))
    if not os.path.exists(lbl_path): continue

    new_img_path = os.path.join(OUTPUT_DIR, 'train/images', img_file)
    new_lbl_path = os.path.join(OUTPUT_DIR, 'train/labels', os.path.basename(lbl_path))
    shutil.copy2(img_path, new_img_path)
    shutil.copy2(lbl_path, new_lbl_path)

    with open(lbl_path) as f:
        lines = f.readlines()
    if not lines: continue

    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    boxes, labels = [], []
    for line in lines:
        cls, *box = map(float, line.strip().split())
        class_counts[int(cls)] += 1
        boxes.append(box)
        labels.append(int(cls))
    data.append((img_file, img_path, boxes, labels, (w, h)))

print("\nBefore Augmentation:")
summarize_distribution(os.path.join(OUTPUT_DIR, 'train/labels'), class_names)
#max_count = max(class_counts.values())
max_count = 100
max_class = max(class_counts, key=class_counts.get)
print(f"\n🔢 Max class count is: {max_count} (Class ID: {max_class} - {class_names[max_class]})")
transform = get_augmentation_pipeline()

print("\nAugmenting minority classes...")
for cls_id in range(len(class_names)):
    if class_counts[cls_id] >= max_count:
        continue
    needed = max_count - class_counts[cls_id]
    samples = [d for d in data if cls_id in d[3]]

    for i in range(needed):
        fname, img_path, boxes, labels, (w, h) = random.choice(samples)
        image = cv2.imread(img_path)

        try:
            augmented = transform(image=image, bboxes=boxes, class_labels=labels)
        except Exception:
            continue

        aug_img = augmented['image']
        aug_boxes = augmented['bboxes']
        aug_labels = augmented['class_labels']

        if not aug_boxes:
            continue

        # Visualize first few
        if i < VISUALIZE_COUNT:
            vis_boxes = [[b[0], b[1], b[2]-b[0], b[3]-b[1]] for b in aug_boxes]  # x_min, y_min, w, h
            visualize(aug_img, vis_boxes, aug_labels, {i: n for i, n in enumerate(class_names)})

        # Save
        aug_name = f"aug_{cls_id}_{i}_{fname}"
        cv2.imwrite(os.path.join(OUTPUT_DIR, 'train/images', aug_name), aug_img)

        with open(os.path.join(OUTPUT_DIR, 'train/labels', aug_name.replace('.jpg', '.txt').replace('.png', '.txt')), 'w') as f:
            for box, label in zip(aug_boxes, aug_labels):
                line = f"{label} " + " ".join([f"{v:.6f}" for v in box])
                f.write(line + '\n')
        class_counts[cls_id] += 1

print("\nAfter Augmentation:")
summarize_distribution(os.path.join(OUTPUT_DIR, 'train/labels'), class_names)
print(f"\n✅ Balanced training set saved to: {OUTPUT_DIR}")
