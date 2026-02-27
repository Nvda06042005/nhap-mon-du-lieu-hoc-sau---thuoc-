"""
Chia ảnh VAIPE thành 3 thư mục: train / val / test
===================================================
- Đọc train_list.csv, test_list.csv
- Bỏ lớp < 10 mẫu
- Tách train → train (80%) + val (20%) stratified
- Copy ảnh vào cấu trúc: dataset/train/<class>/, dataset/val/<class>/, dataset/test/<class>/
"""

import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
from tqdm import tqdm

# ============ CONFIG ============
DATA_DIR = r'd:\projectdulieuhocsau17\archive\pills_data'
OUTPUT_DIR = r'd:\projectdulieuhocsau17\dataset'
MIN_SAMPLES = 10
VAL_RATIO = 0.2
SEED = 42

# ============ LOAD CSV ============
print("Đọc CSV...")
train_df = pd.read_csv(os.path.join(DATA_DIR, 'train_list.csv'))
test_df = pd.read_csv(os.path.join(DATA_DIR, 'test_list.csv'))

print(f"  Gốc: Train={len(train_df)}, Test={len(test_df)}")
print(f"  Classes gốc: {train_df['class'].nunique()}")

# ============ LỌC CLASS ============
all_df = pd.concat([train_df, test_df], ignore_index=True)
class_counts = all_df['class'].value_counts()
valid_classes = set(class_counts[class_counts >= MIN_SAMPLES].index)
removed = set(class_counts.index) - valid_classes

train_df = train_df[train_df['class'].isin(valid_classes)].copy()
test_df = test_df[test_df['class'].isin(valid_classes)].copy()

print(f"  Bỏ {len(removed)} lớp < {MIN_SAMPLES} mẫu: {sorted(removed)}")
print(f"  Giữ {len(valid_classes)} lớp")

# ============ CHIA TRAIN → TRAIN + VAL ============
train_split, val_split = train_test_split(
    train_df, test_size=VAL_RATIO,
    stratify=train_df['class'],
    random_state=SEED
)

print(f"\n  Train: {len(train_split)}")
print(f"  Val:   {len(val_split)}")
print(f"  Test:  {len(test_df)}")
print(f"  Tổng:  {len(train_split) + len(val_split) + len(test_df)}")

# ============ COPY ẢNH ============
def copy_images(df, split_name):
    split_dir = os.path.join(OUTPUT_DIR, split_name)
    copied = 0
    missing = 0
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  {split_name}"):
        src = os.path.join(DATA_DIR, row['path'])
        cls = str(row['class'])
        filename = os.path.basename(row['path'])
        
        dst_dir = os.path.join(split_dir, cls)
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, filename)
        
        if os.path.exists(src):
            shutil.copy2(src, dst)
            copied += 1
        else:
            missing += 1
    
    return copied, missing

# Xóa output cũ
if os.path.exists(OUTPUT_DIR):
    print(f"\nXóa thư mục cũ: {OUTPUT_DIR}")
    shutil.rmtree(OUTPUT_DIR)

print("\nĐang copy ảnh...")
train_copied, train_miss = copy_images(train_split, 'train')
val_copied, val_miss = copy_images(val_split, 'val')
test_copied, test_miss = copy_images(test_df, 'test')

# ============ VERIFY ============
print(f"\n{'='*50}")
print("KẾT QUẢ")
print(f"{'='*50}")
print(f"  Output: {OUTPUT_DIR}")

for split in ['train', 'val', 'test']:
    split_dir = os.path.join(OUTPUT_DIR, split)
    classes = sorted(os.listdir(split_dir))
    total = sum(len(os.listdir(os.path.join(split_dir, c))) for c in classes)
    print(f"  {split:5s}: {total:6d} ảnh, {len(classes)} lớp")

print(f"\n  Cấu trúc:")
print(f"    {OUTPUT_DIR}/")
print(f"    ├── train/<class_id>/*.jpg")
print(f"    ├── val/<class_id>/*.jpg")
print(f"    └── test/<class_id>/*.jpg")

print(f"\n  Missing files: train={train_miss}, val={val_miss}, test={test_miss}")
print("DONE!")
