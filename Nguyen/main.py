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
               #tool_choice="sum"
               )
    
    tools = [sum]
    llm_with_tools = llm.bind_tools(tools)

    tool_map = {tool.name: tool for tool in tools}

    messages = [
        SystemMessage(content="sử dụng tool sum để trả lời câu hỏi"),
        HumanMessage(content="1 cộng 1 bằng bao nhiêu"),
    ]
    
    print("=== Lần gọi đầu tiên ===")
    response = llm_with_tools.invoke(messages)
    print(f"Response: {response}")
    
    if hasattr(response, 'tool_calls') and response.tool_calls:

        messages.append(response)

        for tool_call in response.tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call['args']
            tool_id = tool_call['id']

            if tool_name in tool_map:
                tool = tool_map[tool_name]
                result = tool.invoke(tool_args)
                print(f"Kết quả tool: {result}")
                
                # Add tool result to messages
                tool_message = ToolMessage(
                    content=str(result),
                    tool_call_id=tool_id
                )
                messages.append(tool_message)
        
        # Second call - LLM will use tool results to generate final answer
        print("\n=== Lần gọi thứ hai (với kết quả tool) ===")
        final_response = llm_with_tools.invoke(messages)
        print(f"Câu trả lời cuối cùng: {final_response}")
    else:
        print("Không có tool calls")

if __name__ == "__main__":
    main()