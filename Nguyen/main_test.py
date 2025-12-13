from langchain_vnpt.langchain_vnpt import LangChainVNPT
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from graph.tools import query_vector_store
import json

def get_api_key(llmApiName: str, 
                path      : str = "api-keys.json"):
    with open(path, "r") as f:
        loaded_data = json.load(f)
    for key in loaded_data:
        if key["llmApiName"] == llmApiName:
            return key
from typing_extensions import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
def graph_test(llm):
    class State(TypedDict):
        message: str
        final_answer:Optional[str]

    def llm_node(state:State):
        prompt = """Bạn phải trả về JSON hợp lệ theo định dạng sau:
{"answer": "X", "explain": "lời giải thích ngắn gọn"}

Trong đó:
- answer: chữ cái đáp án đúng (A, B, C, hoặc D)
- explain: giải thích ngắn gọn (tối đa 2-3 câu)

QUAN TRỌNG: 
- Phải đóng chuỗi JSON đầy đủ với dấu ngoặc kép và ngoặc nhọn cuối cùng
- Giữ phần explain ngắn gọn để tránh vượt giới hạn ký tự
- Chỉ trả về JSON, không thêm text khác"""
        response = llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=state.get("message"))])
        print(response)
        state["final_answer"] = json.loads(response.content)
        return state

    graph_builder = StateGraph(State)
    graph_builder.add_node("llm", llm_node)

    graph_builder.add_edge(START, "llm")
    graph_builder.add_edge("llm", END)
    graph = graph_builder.compile()
    return graph

def main():
    key = get_api_key(llmApiName = "LLM large")
    llm = LangChainVNPT(model = "vnptai-hackathon-large",
               authorization = key["authorization"],
               tokenKey = key["tokenKey"],
               tokenId = key["tokenId"],
               temperature = 0.1,
               top_p = 0.1,
               top_k = 10,
               n = 1,
               max_completion_tokens = 500,
               )
    tools = [query_vector_store]
    graph = graph_test(llm)#.bind_tools(tools))
    message = "Ngôi chùa Ba La Mật được khai dựng vào năm nào? A.1886 B.1990 C.1920 D.1930"
    graph_test_input = {"message":message}
    for chunk in graph.stream(graph_test_input):
        print(chunk)

if __name__ == "__main__":
    main()