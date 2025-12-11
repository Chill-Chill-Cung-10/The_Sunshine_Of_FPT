import json
import lancedb
import numpy as np
import pandas as pd
import pyarrow as pa
from vnpt_embedding import VNPTEmbedding
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import os

def get_api_key(llmApiName: str, 
                path      : str = "api-keys.json"):
    with open(path, "r") as f:
        loaded_data = json.load(f)
    for key in loaded_data:
        if key["llmApiName"] == llmApiName:
            return key

def process_batch(embeddings, batch_data, batch_idx, batch_start, batch_end):
    """Xử lý một batch với embedding"""
    try:
        print(f"Worker đang xử lý batch {batch_idx}: dòng {batch_start+1} đến {batch_end}")
        batch_embeddings = embeddings.embed_query(batch_data)
        print(f"Hoàn thành embedding batch {batch_idx}")
        return batch_idx, batch_embeddings, None
    except Exception as e:
        print(f"Lỗi khi xử lý batch {batch_idx}: {str(e)}")
        return batch_idx, None, str(e) 

def main():
    key = get_api_key("LLM embedings")
    embeddings = VNPTEmbedding(authorization = key["authorization"],
                            tokenKey = key["tokenKey"],
                            tokenId = key["tokenId"])

    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    vector_store_path = os.path.join(parent_dir, "vector_store")
    
    db = lancedb.connect(uri=vector_store_path)

    df = pd.read_csv("data.csv")
    
    title = list(df["Title"])
    data = list(df["Data"])
    issue_date = list(df["IssueDate"])
    status = list(df["Status"])
    field = list(df["Field"])

    batch_size = 500
    num_workers = 20
    
    print(f"Tổng số dòng: {len(data)}")
    print(f"Bắt đầu xử lý với batch size = {batch_size} và {num_workers} workers")
    
    schema = pa.schema([
        pa.field("Title", pa.string()),
        pa.field("Data", pa.string()),
        pa.field("IssueDate", pa.string()),
        pa.field("Status", pa.string()),
        pa.field("Field", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), 1024))
    ])
    
    table = None
    total_saved = 0
    lock = threading.Lock()
    
    batches = []
    for i in range(0, len(data), batch_size):
        batch_end = min(i + batch_size, len(data))
        batch_data = data[i:batch_end]
        batches.append((i, batch_end, batch_data, i//batch_size + 1))
    
    completed_batches = {}
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_batch = {
            executor.submit(process_batch, embeddings, batch[2], batch[3], batch[0], batch[1]): batch
            for batch in batches
        }
        
        for future in as_completed(future_to_batch):
            batch_idx, batch_embeddings, error = future.result()
            
            if error:
                print(f"Batch {batch_idx} thất bại: {error}")
                continue
            
            completed_batches[batch_idx] = batch_embeddings

    print("\nĐang lưu dữ liệu vào vector store...")
    for i, (batch_start, batch_end, batch_data, batch_idx) in enumerate(batches):
        if batch_idx not in completed_batches:
            print(f"Bỏ qua batch {batch_idx} do lỗi")
            continue
        
        batch_embeddings = completed_batches[batch_idx]

        batch_embeddings_np = [np.array(emb, dtype=np.float32) for emb in batch_embeddings]
        
        batch_df = pd.DataFrame({
            "Title": title[batch_start:batch_end],
            "Data": data[batch_start:batch_end],
            "IssueDate": issue_date[batch_start:batch_end],
            "Status": status[batch_start:batch_end],
            "Field": field[batch_start:batch_end],
            "vector": batch_embeddings_np
        })
        
        with lock:
            if table is None:
                table = db.create_table("embeddings_data", data=batch_df, schema=schema, mode="overwrite")
                #table.create_index(metric="cosine",
                #                   vector_column_name="vector",
                #                   num_partitions=16384,
                #                   num_sub_vectors=128,
                #                   replace=True)
            else:
                table.add(batch_df)
            
            total_saved += len(batch_df)
            print(f"Đã lưu batch {batch_idx} - Tổng: {total_saved}/{len(data)} bản ghi")
    
    print(f"\nHoàn tất! Đã lưu tổng cộng {total_saved} bản ghi vào LanceDB table 'embeddings_data'")

    
if __name__ == "__main__":
    main()