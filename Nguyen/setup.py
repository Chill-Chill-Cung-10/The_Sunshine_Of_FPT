from langchain_vnpt.langchain_vnpt import LangChainVNPT
from graph.tools import query_vector_store
from graph.graph import create_graph
from embedding.embedding import vector_store_init
import json
import lancedb
import os
from pathlib import Path

_API_KEY_CACHE = {}
_PROMPT_CACHE = {}

def get_api_key(llmApiName: str, path: str = "api-keys.json"):
    """Load API key from JSON file with caching"""
    if llmApiName in _API_KEY_CACHE:
        return _API_KEY_CACHE[llmApiName]
    
    with open(path, "r") as f:
        loaded_data = json.load(f)
    for key in loaded_data:
        if key["llmApiName"] == llmApiName:
            _API_KEY_CACHE[llmApiName] = key
            return key
    return None

def load_prompt(prompt_name: str, prompts_dir: str = "prompts") -> str:
    """Load prompt content from file with caching"""
    if prompt_name in _PROMPT_CACHE:
        return _PROMPT_CACHE[prompt_name]
    
    prompt_path = Path(__file__).parent / prompts_dir / f"{prompt_name}.txt"
    with open(prompt_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    _PROMPT_CACHE[prompt_name] = content
    return content

def setup_vector_store():
    """
    Kiểm tra và khởi tạo vector store nếu chưa tồn tại
    """
    
    vector_store_path = Path(__file__).parent / "vector_store"
    
    if not vector_store_path.exists():
        vector_store_init()
    try:
        db = lancedb.connect(uri=str(vector_store_path))
        table = db.open_table("embeddings_data")
        
        # Warm-up query để cache connection
        print(f"Vector store ready: {table.count_rows()} documents")
        
        return True
    except Exception as e:
        raise RuntimeError(f"Failed to initialize vector store: {e}")

def setup_model_and_graph():
    """
    Khởi tạo LLM và graph - gọi hàm này trước khi bắt đầu tính thời gian
    
    Returns:
        graph: Compiled LangGraph ready for inference
    """
    
    setup_vector_store()
    
    key = get_api_key(llmApiName="LLM small")
    if not key:
        raise ValueError("API key not found")
    
    # Initialize LLM
    llm = LangChainVNPT(
        model="vnptai-hackathon-small",
        authorization=key["authorization"],
        tokenKey=key["tokenKey"],
        tokenId=key["tokenId"],
        temperature=0.1,
        top_p=0.1,
        top_k=10,
        n=1,
        max_completion_tokens=500,
    )
    
    # Load prompts
    external_knowledge_prompt = load_prompt("external_knowledge_prompt")
    reading_comprehension_prompt = load_prompt("reading_comprehension_prompt")
    math_logic_prompt = load_prompt("math_logic_prompt")
    
    # Create graph
    tools = [query_vector_store]
    graph = create_graph(
        llm=llm,
        tools=tools,
        external_knowledge_prompt=external_knowledge_prompt,
        reading_comprehension_prompt=reading_comprehension_prompt,
        math_logic_prompt=math_logic_prompt
    )
    
    return graph

if __name__ == "__main__":
    graph = setup_model_and_graph()
    print("Setup successful! Graph is ready for inference.")
