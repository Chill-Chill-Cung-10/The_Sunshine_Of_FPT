import json
import time
import requests
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- CẤU HÌNH ---
# API Endpoint
API_URL = 'https://api.idg.vnpt.vn/data-service/vnptai-hackathon-embedding'

# Dữ liệu giả lập (Sinh 10,000 câu để test cho chính xác)
base_sentences = [
    "Quyết định về việc ban hành quy chế làm việc.",
    "Thông tư hướng dẫn thi hành luật đất đai.",
    "Nghị định về xử phạt vi phạm hành chính.",
    "Thủ tục cấp đổi hộ chiếu phổ thông trực tuyến.",
    "Hướng dẫn đăng ký khai sinh và bảo hiểm y tế.",
    "Luật Bảo vệ môi trường sửa đổi năm 2024.",
    "Chính sách hỗ trợ doanh nghiệp sau đại dịch.",
    "Công nghệ Blockchain và ứng dụng thực tiễn.",
    "Trí tuệ nhân tạo trong y tế thông minh.",
    "Chuyển đổi số quốc gia giai đoạn 2025."
]
full_dataset = base_sentences * 1000 # 10,000 câu

# --- CLASS GỌI API ---
class VNPTEmbedder:
    def __init__(self, api_key_path="api-keys.json"):
        with open(api_key_path, "r") as f:
            data = json.load(f)
            # Giả sử lấy key đầu tiên hoặc key tên "LLM embedings"
            key = next((k for k in data if k["llmApiName"] == "LLM embedings"), data[0])
            
        self.headers = {
            "Authorization": key["authorization"],
            "Token-id": key["tokenId"],
            "Token-key": key["tokenKey"],
            "Content-Type": "application/json",
        }

    def embed_batch(self, texts):
        if not texts: return 0
        
        json_data = {
            'model': 'vnptai_hackathon_embedding',
            'input': texts,
            'encoding_format': 'float',
        }
        try:
            # Timeout 60s cho mỗi request
            resp = requests.post(API_URL, headers=self.headers, json=json_data, timeout=60)
            if resp.status_code == 200 and 'data' in resp.json():
                return len(texts) # Trả về số lượng thành công
            else:
                return 0 # Lỗi
        except:
            return 0 # Lỗi mạng/timeout

# --- HÀM CHẠY TEST ---
def run_test(embedder, batch_size, max_workers):
    print(f"--- Đang test: Batch Size = {batch_size} | Workers = {max_workers} ---")
    
    # Chia dữ liệu thành các cục (chunks)
    chunks = [full_dataset[i:i + batch_size] for i in range(0, len(full_dataset), batch_size)]
    
    start_time = time.time()
    total_success = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(embedder.embed_batch, chunk) for chunk in chunks]
        
        for future in as_completed(futures):
            total_success += future.result()
            
    duration = time.time() - start_time
    throughput = total_success / duration if duration > 0 else 0
    
    print(f"   -> Thời gian: {duration:.2f}s")
    print(f"   -> Tốc độ: {throughput:.2f} câu/giây")
    return throughput

# --- MAIN ---
def main():
    try:
        embedder = VNPTEmbedder()
    except Exception as e:
        print(f"Lỗi đọc API Key: {e}")
        return

    # DANH SÁCH CÁC CẤU HÌNH MUỐN THỬ
    # (Batch Size, Số Luồng)
    configs = [
        (500, 10),  # Nhẹ nhàng
        (500, 20),  # Tăng tốc
        (500, 40),  # Stress test
        (1000, 10), # Batch to
        (1000, 20), # Batch to + Nhiều luồng
    ]

    best_throughput = 0
    best_config = None

    print("=== BẮT ĐẦU DÒ TÌM CẤU HÌNH TỐI ƯU ===")
    
    for batch, workers in configs:
        t = run_test(embedder, batch, workers)
        if t > best_throughput:
            best_throughput = t
            best_config = (batch, workers)
        time.sleep(2) # Nghỉ chút cho server thở

    print("\n" + "="*40)
    print(f"CẤU HÌNH TỐI ƯU NHẤT: Batch {best_config[0]} - Workers {best_config[1]}")
    print(f"Tốc độ cao nhất: {best_throughput:.2f} câu/giây")
    
    hours = (100_000_000 / best_throughput) / 3600
    print(f"Thời gian dự kiến cho 100tr dòng: {hours:.2f} giờ")
    print("="*40)

if __name__ == "__main__":
    main()