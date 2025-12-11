from langchain_core.tools import tool
import lancedb
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from embedding.vnpt_embedding import VNPTEmbedding

_API_KEY_CACHE = None
_EMBEDDING_CACHE = None

def get_api_key(llmApiName: str, path: str = "api-keys.json"):
    """Load API key from JSON file with caching"""
    global _API_KEY_CACHE
    
    if _API_KEY_CACHE is not None:
        return _API_KEY_CACHE
    
    with open(path, "r") as f:
        loaded_data = json.load(f)
    for key in loaded_data:
        if key["llmApiName"] == llmApiName:
            _API_KEY_CACHE = key
            return key
    return None

def get_embedding_instance():
    """Get or create cached embedding instance"""
    global _EMBEDDING_CACHE
    
    if _EMBEDDING_CACHE is None:
        key = get_api_key("LLM embedings")
        if not key:
            return None
        
        _EMBEDDING_CACHE = VNPTEmbedding(
            authorization=key["authorization"],
            tokenKey=key["tokenKey"],
            tokenId=key["tokenId"]
        )
    
    return _EMBEDDING_CACHE

@tool("vector store")
def query_vector_store(query: str, top_k: int = 5):
    """
    Query the LanceDB vector store to find relevant documents based on semantic similarity.
    
    :param query: The search query text to find similar documents
    :return: List of relevant documents with their metadata and similarity scores
    """
    try:
        embeddings = get_embedding_instance()
        if not embeddings:
            return {"error": "API key not found"}
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        vector_store_path = os.path.join(parent_dir, "vector_store")
        
        db = lancedb.connect(uri=vector_store_path)
        table = db.open_table("embeddings_data")
        
        query_embedding = embeddings.embed_query([query])[0]

        results = table.search(query_embedding, 
                               vector_column_name="vector"
                            ).distance_type("cosine").limit(top_k).to_list()
        
        formatted_results = []
        for i, result in enumerate(results):
            distance = result.get("_distance", 0)
            if distance <= 0.5:
                formatted_results.append({
                    "rank": len(formatted_results) + 1,
                    "title": result.get("Title", ""),
                    "data": result.get("Data", ""),
                    "issue_date": result.get("IssueDate", ""),
                    "status": result.get("Status", ""),
                    "field": result.get("Field", ""),
                    "distance": distance
                })
        
        return {
            "query": query,
            "top_k": top_k,
            "results": formatted_results
        }
        
    except Exception as e:
        return {"error": str(e)}
    
if __name__ == "__main__":
    print(query_vector_store("Moi anh ve bac ninh em choi tham"))