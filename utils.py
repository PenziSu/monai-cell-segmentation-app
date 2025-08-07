# utils.py
import cv2
import numpy as np
import json
from pathlib import Path
import shutil

def create_visualization(original_img_bytes, label_mask_array):
    """將原始影像和標籤圖融合成一張視覺化圖片"""
    
    original_img = cv2.imdecode(np.frombuffer(original_img_bytes, np.uint8), cv2.IMREAD_COLOR)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    # --- 新增的修正：檢查並縮放標籤圖尺寸 ---
    # 獲取目標尺寸 (來自原始影像)
    target_height, target_width = original_img.shape[:2]
    mask_height, mask_width = label_mask_array.shape

    # 如果尺寸不匹配
    if target_height != mask_height or target_width != mask_width:
        print(f"尺寸不匹配，正在縮放標籤圖：從 ({mask_height}, {mask_width}) -> ({target_height}, {target_width})")
        # 使用「最近鄰插值」來縮放標籤圖，以保持整數標籤的完整性
        label_mask_array = cv2.resize(
            label_mask_array, 
            (target_width, target_height), 
            interpolation=cv2.INTER_NEAREST
        )
    # ------------------------------------

    # 建立彩色遮罩
    color_overlay = np.zeros_like(original_img)
    instance_ids = np.unique(label_mask_array)[1:] # 忽略背景 0

    for instance_id in instance_ids:
        color = np.random.randint(50, 256, size=3, dtype=np.uint8)
        color_overlay[label_mask_array == instance_id] = color.tolist()

    # 影像融合
    blended_img = cv2.addWeighted(original_img, 0.7, color_overlay, 0.3, 0)
    
    return blended_img

def generate_meta_json(mask_array):
    """根據標籤圖產生 meta.json 的內容"""
    contours = int(np.max(mask_array))
    meta_data = {
        "image_size": [mask_array.shape[0], mask_array.shape[1]],
        "contours": contours
    }
    return json.dumps(meta_data, indent=2).encode('utf-8')


def clear_directory(directory_path: Path):
    """
    Clears all contents (files and subdirectories) from a given directory,
    leaving the directory itself intact.
    """
    if not directory_path.is_dir():
        raise ValueError(f"'{directory_path}' is not a directory.")

    for item in directory_path.iterdir():
        if item.is_file():
            item.unlink()  # Remove files
        elif item.is_dir():
            shutil.rmtree(item)  # Recursively remove subdirectories

def is_directory_empty_list(directory_path):
    """Checks if a directory is empty by converting iterdir() to a list."""
    path = Path(directory_path)
    if not path.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory_path}")
    return not list(path.iterdir())