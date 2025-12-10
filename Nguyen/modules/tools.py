from langchain_core.tools import tool
import lancedb
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from embedding.vnpt_embedding import VNPTEmbedding

def get_api_key(llmApiName: str, path: str = "api-keys.json"):
    """Load API key from JSON file"""
    with open(path, "r") as f:
        loaded_data = json.load(f)
    for key in loaded_data:
        if key["llmApiName"] == llmApiName:
            return key
    return None

@tool("vector store")
def query_vector_store(query: str, top_k: int = 5):
    """
    Query the LanceDB vector store to find relevant documents based on semantic similarity.
    
    :param query: The search query text to find similar documents
    :param top_k: Number of top results to return (default: 5)
    :return: List of relevant documents with their metadata and similarity scores
    """
    try:
        key = get_api_key("LLM embedings")
        if not key:
            return {"error": "API key not found"}
        
        embeddings = VNPTEmbedding(
            authorization=key["authorization"],
            tokenKey=key["tokenKey"],
            tokenId=key["tokenId"]
        )
        
        db = lancedb.connect(uri="vector_store")
        table = db.open_table("embeddings_data")
        
        query_embedding = embeddings.embed_query([query])[0]
        
        results = table.search(query_embedding).limit(top_k).to_list()
        
        formatted_results = []
        for i, result in enumerate(results):
            formatted_results.append({
                "rank": i + 1,
                "title": result.get("Title", ""),
                "data": result.get("Data", ""),
                "issue_date": result.get("IssueDate", ""),
                "status": result.get("Status", ""),
                "field": result.get("Field", ""),
                "distance": result.get("_distance", 0)
            })
        
        return {
            "query": query,
            "top_k": top_k,
            "results": formatted_results
        }
        
    except Exception as e:
        return {"error": str(e)}