# BASE IMAGE
# Lưu ý: Sử dụng đúng phiên bản CUDA 12.2 để khớp với Server BTC 
# ------------------------------------------------------------ 
FROM nvidia/cuda:12.2.0-devel-ubuntu20.04 
 
# ------------------------------------------------------------ 
# SYSTEM DEPENDENCIES 
# Cài đặt Python, Pip và các gói hệ thống cần thiết 
# ------------------------------------------------------------ 
RUN apt-get update && apt-get install -y \ 
   python3 \ 
   python3-pip \ 
   git \ 
   && rm -rf /var/lib/apt/lists/* 
 
# Link python3 thành python nếu cần 
RUN ln -s /usr/bin/python3 /usr/bin/python 
 
# ------------------------------------------------------------ 
# PROJECT SETUP 
# ------------------------------------------------------------ 
# Thiết lập thư mục làm việc 
WORKDIR /code 
 
# Copy requirements.txt trước để tận dụng Docker cache
COPY requirements.txt /code/requirements.txt
 
# ------------------------------------------------------------ 
# INSTALL LIBRARIES 
# ------------------------------------------------------------ 
# Nâng cấp pip và cài đặt các thư viện từ requirements.txt 
RUN pip3 install --no-cache-dir --upgrade pip && \ 
   pip3 install --no-cache-dir -r requirements.txt 

# Copy toàn bộ source code vào trong container
# Bao gồm: setup.py, predict.py, graph/, langchain_vnpt/, embedding/, prompts/, vector_store/
COPY . /code/
 
# ------------------------------------------------------------ 
# EXECUTION 
# Lệnh chạy mặc định khi container khởi động
# Pipeline sẽ đọc private_test.json và xuất ra submission.csv, submission_time.csv 
# ------------------------------------------------------------ 
CMD ["bash", "inference.sh"]