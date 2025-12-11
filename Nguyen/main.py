from langchain_vnpt.langchain_vnpt import LangChainVNPT
from graph.tools import query_vector_store
from graph.graph import create_graph
import json
from pathlib import Path

def get_api_key(llmApiName: str, 
                path      : str = "api-keys.json"):
    with open(path, "r") as f:
        loaded_data = json.load(f)
    for key in loaded_data:
        if key["llmApiName"] == llmApiName:
            return key

def load_prompt(prompt_name: str, prompts_dir: str = "prompts") -> str:
    prompt_path = Path(__file__).parent / prompts_dir / f"{prompt_name}.txt"
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()

def load_dataset(split: str = "val"):
    data_path = Path(__file__).parent.parent / "data" / f"{split}.json"
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)

def format_message(item: dict) -> str:
    question_text = item.get("question", "").strip()
    choices = item.get("choices", [])
    
    parts = [question_text]
    for idx, choice in enumerate(choices):
        letter = chr(ord("A") + idx)
        parts.append(f"{letter}. {choice}")
    
    return "\n".join(parts)

def main():
    key = get_api_key(llmApiName = "LLM small")
    llm = LangChainVNPT(model = "vnptai-hackathon-small",
               authorization = key["authorization"],
               tokenKey = key["tokenKey"],
               tokenId = key["tokenId"],
               temperature = 0.1,
               top_p = 0.1,
               top_k = 10,
               n = 1,
               max_completion_tokens = 500,
               )
    external_knowledge_prompt = load_prompt("external_knowledge_prompt")
    reading_comprehension_prompt = load_prompt("reading_comprehension_prompt")
    math_logic_prompt = load_prompt("math_logic_prompt")
    tools = [query_vector_store]
    graph = create_graph(llm=llm,
                 tools=tools,
                 external_knowledge_prompt=external_knowledge_prompt,
                 reading_comprehension_prompt=reading_comprehension_prompt,
                 math_logic_prompt=math_logic_prompt)
    
    split = "val"
    dataset = load_dataset(split)
    
    print(f"Processing{len(dataset)} question from {split} set...")
    
    results_path = Path(__file__).parent / "results.csv"
    if results_path.exists():
        results_path.unlink()
    
    for idx, item in enumerate(dataset, 1):
        qid = item.get("qid")
        message = format_message(item)
        
        print(f"\n[{idx}/{len(dataset)}] Processing {qid}...")
        
        graph_input = {
            "qid": qid,
            "message": message
        }
        
        try:
            result = graph.invoke(graph_input)
            print(f"{qid} completed")
        except Exception as e:
            print(f"{qid} failed: {e}")
    
    print(f"\nCompleted! Results are wrote at {results_path}")

if __name__ == "__main__":
    main()