# 下載docker image
docker pull penzisu/monai-cell-segmentation-app:1.1

# 原始碼建立image
docker build -t monai-cell-segmentation-app .

# 建立container
docker run -d -p 8080:8080 --gpus all --shm-size="2g" monai-cell-segmentation-app

# 直接執行 app.py
streamlit run app.py --server.port=8080 --server.enableCORS=false --server.address=127.0.0.1 --server.maxUploadSize=4096
