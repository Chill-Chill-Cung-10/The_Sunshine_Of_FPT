from modules.message_processer import get_message_by_qid
from modules.VNPT import LangChainVNPT
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_experimental.tools import PythonREPLTool
from modules.tools import sum
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
               )
    
    tools = [sum]
    llm_with_tools = llm.bind_tools(tools)

    messages = [
        SystemMessage(content="sử dụng tool sum (nếu cần) để trả lời câu hỏi"),
        HumanMessage(content="1 cộng 1 bằng bao nhiêu"),
    ]
    
    response = llm_with_tools.invoke(messages)
    print(json.dumps(response.response_metadata.get('api_response'), indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()