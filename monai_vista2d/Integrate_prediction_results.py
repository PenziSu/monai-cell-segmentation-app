import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- 1. 讀取影像 ---
# 讀取原始影像
original_img = cv2.imread(r"D:\nomai_vista2d\monai_vista2d_v0.3.0\cellpose_dataset\test\000_img.png")
# OpenCV 讀取彩色圖像是 BGR 順序，我們轉成習慣的 RGB
original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

# 讀取 TIF 標籤圖 (分割結果)
# cv2.IMREAD_UNCHANGED 確保能讀取到原始的整數資料類型
label_mask = cv2.imread(r"D:\nomai_vista2d\monai_vista2d_v0.3.0\eval\001_img_trans.tif", cv2.IMREAD_UNCHANGED)

print(f"原始影像尺寸: {original_img.shape}")
print(f"標籤圖尺寸: {label_mask.shape}")
print(f"偵測到的細胞實例數量 (最大標籤值): {np.max(label_mask)}")

# --- 2. 建立彩色遮罩 ---
# 建立一個和原圖一樣大的全黑彩色圖層
color_overlay = np.zeros_like(original_img)

# 獲取所有細胞的唯一ID (從 1 開始)
instance_ids = np.unique(label_mask)[1:] # [1:] 用來忽略背景的 0

# --- 3. 上色 ---
for instance_id in instance_ids:
    # 為每個細胞實例產生一個隨機顏色
    color = np.random.randint(50, 256, size=3, dtype=np.uint8)
    
    # 找到這個細胞的所有像素位置
    # np.where(label_mask == instance_id) 會回傳 (y座標列表, x座標列表)
    color_overlay[label_mask == instance_id] = color

# --- 4. 影像融合 (圖層堆疊) ---
# alpha 是原圖權重，beta 是遮罩圖層權重
alpha = 0.7 
beta = 0.3
gamma = 0 # 一個常數項，這裡設為0

# 使用 OpenCV 的 addWeighted 函式進行融合
blended_img = cv2.addWeighted(original_img, alpha, color_overlay, beta, gamma)

# --- 5. 顯示結果 ---
plt.figure(figsize=(15, 7))

plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(original_img)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Instance Segmentation Mask")
# 用一個色彩映射表來顯示標籤圖，不同的數值會有不同的顏色
plt.imshow(label_mask, cmap='nipy_spectral') 
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Blended Result")
plt.imshow(blended_img)
plt.axis('off')

plt.tight_layout()
plt.show()

# 如果要儲存圖片
# Matplotlib 儲存時需要轉回 BGR
blended_img_bgr = cv2.cvtColor(blended_img, cv2.COLOR_RGB2BGR)
cv2.imwrite(r"D:\nomai_vista2d\monai_vista2d_v0.3.0\eval\result_visualization.jpg", blended_img_bgr)
