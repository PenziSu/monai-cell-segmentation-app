# app.py
import streamlit as st
from PIL import Image
import io # 需要 io 模組來處理 bytes
# --- 關鍵修改：解除 Pillow 的像素上限 ---
Image.MAX_IMAGE_PIXELS = None
import subprocess
from pathlib import Path
import imageio.v3 as imageio
import numpy as np
import json
import cv2
import logging
import sys

# 匯入我們自己的工具函式
import utils

# --- 設定 Logging ---

# 1. 建立一個 logger 物件
logger = logging.getLogger()
logger.setLevel(logging.INFO) # 設定 logger 的最低級別

# 2. 建立一個 formatter 物件來定義日誌格式
formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s",
    "%Y-%m-%d %H:%M:%S" # 優化時間格式
)

# 3. 建立 Console Handler (輸出到畫面)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# 4. 建立 File Handler (儲存在檔案)
# RotatingFileHandler 是更好的選擇，它會在檔案太大時自動建立新檔案
from logging.handlers import RotatingFileHandler
file_handler = RotatingFileHandler(
    'app.log',          # 檔案名稱
    maxBytes=10*1024*1024, # 檔案最大 10 MB
    backupCount=5,      # 保留最近 5 個檔案
    encoding='utf-8'
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# --- Logging 設定結束 ---


# --- 頁面設定 ---
st.set_page_config(layout="wide", page_title="NVIDIA MONAI Vista2D 細胞分割")

# --- 常數設定 ---
# 建立必要的資料夾
UPLOADS_DIR = Path("images")
UPLOADS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR = Path("eval")
OUTPUTS_DIR.mkdir(exist_ok=True)

# 定義固定的檔案路徑
SOURCE_IMAGE_PATH = UPLOADS_DIR / "source_image.png"
CONFIG_FILE_PATH = "monai_vista2d/configs/inference.json"
RESULT_TIF_PATH = OUTPUTS_DIR / "source_image_label.tif"

# --- Streamlit 頁面佈局 ---
st.title("NVIDIA MONAI Vista2D 細胞分割預測")

# 依照你的設計圖，側邊欄是空的
# st.sidebar.title("...")

uploaded_file = st.file_uploader("選擇一張影像檔案", type=["png", "jpg", "jpeg", "tif", "tiff"])

if uploaded_file is not None:    
    original_name = str(uploaded_file.name).split(".")[0]
    bytes_data = uploaded_file.getvalue()

    if "last_uploaded_filename" not in st.session_state or st.session_state.last_uploaded_filename != uploaded_file.name:
        st.session_state.inference_done = False
        st.session_state.error = None
        st.session_state.label_ready = False
        st.session_state.last_uploaded_filename = uploaded_file.name
        logging.info(f"收到新上傳的檔案: {uploaded_file.name}")

    # 將上傳的檔案儲存到固定路徑
    # 注意：我們將原始 TIF 儲存為 PNG 格式以簡化後續處理，MONAI bundle 會處理它
    # 如果 MONAI bundle 需要 TIF 格式，則應儲存為 .tif
    pil_image_to_save = Image.open(io.BytesIO(bytes_data))
    pil_image_to_save.save(SOURCE_IMAGE_PATH, "PNG")


    # --- 關鍵修改 Part 2：產生並顯示預覽圖 ---
    with st.spinner("正在產生預覽圖..."):
        preview_img = Image.open(io.BytesIO(bytes_data))
        # 建立一個最大為 512x512 的預覽圖，同時保持原始長寬比
        preview_img.thumbnail((512, 512))
        st.image(preview_img, caption="上傳原始影像的預覽圖")
    # -------------------------------------------

    if st.button("開始推論"):
        st.session_state.inference_done = False # 重新推論
        
        with st.spinner("模型推論中，請稍候..."):
            logging.info("開始執行 MONAI Bundle 推論...")
            try:
                command = [
                    "python", "-m", "monai.bundle", "run",
                    "--config_file", CONFIG_FILE_PATH
                ]                
                result = subprocess.run(command, capture_output=True, text=True, check=True)
                st.session_state.log_output = result.stdout
                st.session_state.inference_done = True
                st.session_state.error = None                
                logging.info("推論程序成功完成。")
            except subprocess.CalledProcessError as e:
                logging.error("推論程序執行失敗！")
                # 如果失敗，同時儲存 stdout 和 stderr 以便除錯
                log_output = "--- STDOUT ---\n" + e.stdout + "\n\n--- STDERR ---\n" + e.stderr
                st.session_state.log_output = log_output
                st.session_state.inference_done = False
                st.session_state.error = e.stderr
            except FileNotFoundError:
                st.session_state.inference_done = False
                st.session_state.error = "錯誤：找不到 'python' 指令。請確認環境設定正確。"


# --- 顯示結果與下載按鈕 ---
if st.session_state.get('inference_done'):
    st.success("推論成功！")

    # --- 新增：顯示可摺疊的 Log 區塊 ---
    if st.session_state.get("log_output"):
        with st.expander("點此查看詳細執行日誌"):
            st.code(st.session_state.log_output)
    # ------------------------------------

    
    # 注意：這裡的 bytes_data 是原始高解析度影像的 bytes
    # 我們需要重新讀取它來進行視覺化
    with open(SOURCE_IMAGE_PATH, "rb") as f:
        original_image_bytes_for_viz = f.read()
    label_mask_array = imageio.imread(RESULT_TIF_PATH)

    # with st.spinner("疊加影像中，請稍候..."):
        # blended_image = utils.create_visualization(original_image_bytes_for_viz, label_mask_array)
    
    # 顯示疊加後的結果圖
    # st.image(blended_image, caption="預測結果（原始影像 + 遮罩）")
    
    st.markdown("---")
    st.header("下載檔案")
    if st.session_state.label_ready == False:
        logging.info("檔案完成，可下載。")
        st.session_state.label_ready = True

    # 準備下載檔案
    # 1. 推論結果 TIF
    with open(RESULT_TIF_PATH, "rb") as f:
        st.download_button(
            label="下載 TIF 標籤圖",
            data=f,
            file_name=f"{original_name}_mask.tif",
            mime="image/tiff"
        )
        
    # 2. meta.json
    meta_json_bytes = utils.generate_meta_json(label_mask_array)
    st.download_button(
        label="下載 Meta.json",
        data=meta_json_bytes,
        file_name=f"{original_name}_meta.json",
        mime="application/json"
    )
    
    # 3. 視覺化 JPG
    # 需要先將 numpy array 轉為可供下載的 bytes
    # is_success, buffer = cv2.imencode(".jpg", cv2.cvtColor(blended_image, cv2.COLOR_RGB2BGR))
    # if is_success:
    #     st.download_button(
    #         label="下載視覺化 JPG",
    #         data=buffer.tobytes(),
    #         file_name="visualization.jpg",
    #         mime="image/jpeg"
    #     )

elif st.session_state.get('error'):
    st.error("推論過程中發生錯誤：")
    st.code(st.session_state.error)