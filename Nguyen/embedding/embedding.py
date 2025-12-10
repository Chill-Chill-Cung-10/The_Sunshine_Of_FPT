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

def csv_process():
    df = pd.read_csv("data.csv")
    title = list(df["Title"])
    data = list(df["data"])
    issue_date = list(df["IssueDate"])
    status = list(df["Status"])
    field = list(df["Field"])

    documents = []
    
    return documents

def main():
    key = get_api_key("LLM embedings")
    embeddings = VNPTEmbedding(authorization = key["authorization"],
                            tokenKey = key["tokenKey"],
                            tokenId = key["tokenId"])

    schema = pa.schema([
    pa.field("vector", pa.list_(pa.float32(), 1024)),
    pa.field("title", pa.string()),
    pa.field("content", pa.string()),
    pa.field("issue_date", pa.string()),
    pa.field("status", pa.string()),
    pa.field("field", pa.string()),
])

    print(embeddings.embed_query(["Hello", "dsfdo"]))

if __name__ == "__main__":
    main()