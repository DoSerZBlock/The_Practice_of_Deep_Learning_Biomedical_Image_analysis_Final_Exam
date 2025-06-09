
import os
import shutil
import random
from sklearn.model_selection import train_test_split
import pandas as pd
from PIL import Image
import numpy as np

def create_directory_structure(base_dir):
    """建立資料目錄結構"""
    splits = ['train', 'val', 'test']
    classes = ['benign', 'malignant', 'normal']

    for split in splits:
        for class_name in classes:
            dir_path = os.path.join(base_dir, split, class_name)
            os.makedirs(dir_path, exist_ok=True)
    print("✓ 資料目錄結構已建立")

def split_dataset(source_dir, target_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """分割BUSI資料集為訓練、驗證和測試集"""

    # 設定隨機種子
    random.seed(42)

    classes = ['benign', 'malignant', 'normal']

    for class_name in classes:
        class_dir = os.path.join(source_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"警告：找不到 {class_dir} 目錄")
            continue

        # 取得所有影像檔案
        images = [f for f in os.listdir(class_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # 分割資料
        train_imgs, temp_imgs = train_test_split(images, test_size=(val_ratio + test_ratio), 
                                                random_state=42)
        val_imgs, test_imgs = train_test_split(temp_imgs, 
                                             test_size=test_ratio/(val_ratio + test_ratio), 
                                             random_state=42)

        # 複製檔案到對應目錄
        splits = {'train': train_imgs, 'val': val_imgs, 'test': test_imgs}

        for split, img_list in splits.items():
            for img in img_list:
                src_path = os.path.join(class_dir, img)
                dst_path = os.path.join(target_dir, split, class_name, img)
                shutil.copy2(src_path, dst_path)

        print(f"{class_name}: 訓練集 {len(train_imgs)}, 驗證集 {len(val_imgs)}, 測試集 {len(test_imgs)}")

def analyze_dataset(data_dir):
    """分析資料集統計資訊"""
    statistics = {}

    for split in ['train', 'val', 'test']:
        split_stats = {}
        split_dir = os.path.join(data_dir, split)

        if os.path.exists(split_dir):
            for class_name in ['benign', 'malignant', 'normal']:
                class_dir = os.path.join(split_dir, class_name)
                if os.path.exists(class_dir):
                    count = len([f for f in os.listdir(class_dir) 
                               if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                    split_stats[class_name] = count
                else:
                    split_stats[class_name] = 0

        statistics[split] = split_stats

    return statistics

def check_image_quality(data_dir):
    """檢查影像品質和格式"""
    issues = []

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(root, file)
                try:
                    with Image.open(file_path) as img:
                        # 檢查影像大小
                        if img.size[0] < 100 or img.size[1] < 100:
                            issues.append(f"小影像: {file_path} - {img.size}")

                        # 檢查色彩模式
                        if img.mode not in ['RGB', 'L']:
                            issues.append(f"異常色彩模式: {file_path} - {img.mode}")

                except Exception as e:
                    issues.append(f"無法開啟: {file_path} - {str(e)}")

    return issues

if __name__ == "__main__":
    # 設定路徑
    source_dir = "data/original_BUSI"  # 原始BUSI資料集路徑
    target_dir = "data/BUSI_dataset"   # 處理後的資料集路徑

    # 建立目錄結構
    create_directory_structure(target_dir)

    # 分割資料集
    print("開始分割資料集...")
    # split_dataset(source_dir, target_dir)

    # 分析資料集
    # stats = analyze_dataset(target_dir)
    # print("\n資料集統計:")
    # for split, counts in stats.items():
    #     total = sum(counts.values())
    #     print(f"{split}: {counts} (總計: {total})")

    # 檢查影像品質
    # issues = check_image_quality(target_dir)
    # if issues:
    #     print(f"\n發現 {len(issues)} 個影像問題:")
    #     for issue in issues[:10]:  # 只顯示前10個問題
    #         print(f"  {issue}")
    # else:
    #     print("\n✓ 所有影像檢查通過")
