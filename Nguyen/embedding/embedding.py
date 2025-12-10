import json
import lancedb
import pandas as pd
import pyarrow as pa
from vnpt_embedding import VNPTEmbedding

def get_api_key(llmApiName: str, 
                path      : str = "api-keys.json"):
    with open(path, "r") as f:
        loaded_data = json.load(f)
    for key in loaded_data:
        if key["llmApiName"] == llmApiName:
            return key 

def main():
    key = get_api_key("LLM embedings")
    embeddings = VNPTEmbedding(authorization = key["authorization"],
                            tokenKey = key["tokenKey"],
                            tokenId = key["tokenId"])

    db = lancedb.connect(uri="vector_store")

    df = pd.read_csv("data.csv")
    
    title = list(df["Title"])
    data = list(df["Data"])
    issue_date = list(df["IssueDate"])
    status = list(df["Status"])
    field = list(df["Field"])

    batch_size = 500
    
    print(f"Tổng số dòng: {len(data)}")
    print(f"Bắt đầu xử lý với batch size = {batch_size}")
    
    schema = pa.schema([
        pa.field("Title", pa.string()),
        pa.field("Data", pa.string()),
        pa.field("IssueDate", pa.string()),
        pa.field("Status", pa.string()),
        pa.field("Field", pa.string()),
        pa.field("vector", pa.list_(pa.float32()))
    ])
    
    table = None
    total_saved = 0
    
    for i in range(0, len(data), batch_size):
        batch_end = min(i + batch_size, len(data))
        batch_data = data[i:batch_end]
        
        print(f"Đang xử lý batch {i//batch_size + 1}: dòng {i+1} đến {batch_end}")
        
        batch_embeddings = embeddings.embed_query(batch_data)
        
        batch_df = pd.DataFrame({
            "Title": title[i:batch_end],
            "Data": data[i:batch_end],
            "IssueDate": issue_date[i:batch_end],
            "Status": status[i:batch_end],
            "Field": field[i:batch_end],
            "vector": batch_embeddings
        })
        
        if table is None:
            table = db.create_table("embeddings_data", data=batch_df, schema=schema, mode="overwrite")
        else:
            table.add(batch_df)
        
        total_saved += len(batch_df)
        print(f"Hoàn thành batch {i//batch_size + 1} - Đã lưu {total_saved}/{len(data)} bản ghi vào vector store")
    
    print(f"Hoàn tất! Đã lưu tổng cộng {total_saved} bản ghi vào LanceDB table 'embeddings_data'")

    
if __name__ == "__main__":
    main()