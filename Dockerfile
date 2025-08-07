# --- 基礎層：選擇一個包含 Python 的官方映像 ---
# 使用 Python 3.10 的輕量版作為基礎
FROM python:3.10-slim

# --- 設定工作目錄 ---
# 在映像中建立一個 /app 資料夾，並將其設為後續指令的工作目錄
WORKDIR /app

# --- 安裝系統層級的相依套件 (如果需要) ---
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
 && rm -rf /var/lib/apt/lists/*

# --- 安裝 Python 相依套件 ---
# 1. 先只複製 requirements.txt 進來
COPY requirements.txt .

# 2. 安裝 PyTorch (GPU 版本) - 這是最關鍵的一步
# 我們使用 --index-url 來指定安裝 CUDA 11.8 的版本
RUN pip install torch==2.6.0 torchvision==0.21.0 --extra-index-url https://download.pytorch.org/whl/cu118

# 3. 安裝 requirements.txt 中剩餘的套件
# 我們在 torch 安裝後再安裝其他套件，避免版本被覆蓋
RUN pip install --no-cache-dir -r requirements.txt

# --- 複製應用程式碼 ---
# 將本地所有檔案 (monai_vista2d/, app.py, utils.py) 複製到映像的 /app/ 目錄下
COPY . .

# --- 設定環境變數 (可選) ---
# ENV NVIDIA_VISIBLE_DEVICES all
# ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# --- 開放 Port ---
# 聲明我們的 Streamlit 應用會使用 8080 port
EXPOSE 8080

# --- 啟動指令 ---
# 設定當容器啟動時，預設要執行的指令
# 啟動 Streamlit，並指定 port 為 8080，且允許從外部訪問
# streamlit run app.py --server.port=8080 --server.enableCORS=false --server.address=0.0.0.0 --server.maxUploadSize=4096
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.enableCORS=false", "--server.address=0.0.0.0", "--server.maxUploadSize=4096"]