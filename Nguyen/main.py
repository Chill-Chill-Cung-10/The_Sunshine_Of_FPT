from modules.message_processer import get_message_by_qid
from modules.VNPT import LangChainVNPT
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_experimental.tools import PythonREPLTool
import json

def get_api_key(llmApiName: str, 
                path      : str = "api-keys.json"):
    with open(path, "r") as f:
        loaded_data = json.load(f)
    for key in loaded_data:
        if key["llmApiName"] == llmApiName:
            return key

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
               max_completion_tokens = 300,
               #tool_choice="sum"
               )
    
    python_tool = PythonREPLTool()
    tools = [python_tool]

    llm = llm.bind_tools(tools)

    prompt = """Chỉ cần trả về câu trả lời đúng (A,B,..). {"answer": "E", "explain": "explaination"}Trong đó answer là câu trả lời đúng, explain là lời giải thích cho answer, explain phải có dẫn chứng đầy đủ."""
    question = get_message_by_qid("val_0032", split="val")
    response = llm.invoke([
        SystemMessage(content="sử dụng tool sum để trả lời câu hỏi"),
        HumanMessage(content="1 cộng 1 bằng bao nhiêu"),
    ])

    print(response)

if __name__ == "__main__":
    main()