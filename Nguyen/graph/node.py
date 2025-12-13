from .state import State
from langchain_vnpt.langchain_vnpt import LangChainVNPT
from langchain_core.messages import HumanMessage, SystemMessage
from .tools import query_vector_store
import json

def rag(state: State):
    """Query vector store and add context to state if available"""
    result = query_vector_store(query=state.get("message"), top_k=3)

    if "results" in result and result["results"]:
        context_parts = []
        for item in result["results"]:
            context_parts.append(f"- {item['data']}")
        
        if context_parts:
            state["rag_context"] = "\n".join(context_parts)
    
    return state

def classifier(state:State, svm_model):
    """
    Classify câu hỏi thành 3 loại:
    0: external_knowledge
    1: reading_comprehension
    2: math_logic
    """
    question = state.get("message")
    
    # Predict category
    prediction = svm_model.predict([question])[0]
    
    # Map prediction to route
    category_map = {
        0: "reading_comprehension",
        1: "math_logic",
        2: "external_knowledge"
    }
    
    state["category"] = category_map.get(prediction, "external_knowledge")
    
    return state

def external_knowledge(state:State, 
                       llm:LangChainVNPT,
                       prompt:str):
    rag_context = state.get("rag_context", None)
    
    if rag_context:
        enhanced_prompt = f"{prompt}\n\nThông tin tham khảo từ cơ sở dữ liệu:\n{rag_context}"
    else:
        enhanced_prompt = prompt
    
    response = llm.invoke(
        [SystemMessage(content=enhanced_prompt),
         HumanMessage(content=state.get("message"))]
    )
    try:
        state["answer"] = json.loads(response.content)
    except json.JSONDecodeError as e:
        try:
            if '"answer"' in response.content:
                import re
                match = re.search(r'"answer"\s*:\s*"([A-Z])"', response.content)
                if match:
                    state["answer"] = {"answer": match.group(1), "explain": "Fallback due to JSON parse error"}
                else:
                    state["answer"] = {"answer": "", "explain": f"JSON parse error: {str(e)}"}
            else:
                state["answer"] = {"answer": "", "explain": f"JSON parse error: {str(e)}"}
        except Exception:
            state["answer"] = {"answer": "", "explain": "Failed to parse response"}
    return state

def reading_comprehension(state:State,
                          llm:LangChainVNPT,
                          prompt:str):
    response = llm.invoke([SystemMessage(content=prompt),
                          HumanMessage(content=state.get("message"))
                        ])
    try:
        state["answer"] = json.loads(response.content)
    except json.JSONDecodeError as e:
        try:
            if '"answer"' in response.content:
                import re
                match = re.search(r'"answer"\s*:\s*"([A-Z])"', response.content)
                if match:
                    state["answer"] = {"answer": match.group(1), "explain": "Fallback due to JSON parse error"}
                else:
                    state["answer"] = {"answer": "", "explain": f"JSON parse error: {str(e)}"}
            else:
                state["answer"] = {"answer": "", "explain": f"JSON parse error: {str(e)}"}
        except Exception:
            state["answer"] = {"answer": "", "explain": "Failed to parse response"}
    return state

def math_logic(state:State,
               llm:LangChainVNPT,
               prompt:str):
    response = llm.invoke([SystemMessage(content=prompt),
                          HumanMessage(content=state.get("message"))
                        ])
    try:
        state["answer"] = json.loads(response.content)
    except json.JSONDecodeError as e:
        try:
            if '"answer"' in response.content:
                import re
                match = re.search(r'"answer"\s*:\s*"([A-Z])"', response.content)
                if match:
                    state["answer"] = {"answer": match.group(1), "explain": "Fallback due to JSON parse error"}
                else:
                    state["answer"] = {"answer": "", "explain": f"JSON parse error: {str(e)}"}
            else:
                state["answer"] = {"answer": "", "explain": f"JSON parse error: {str(e)}"}
        except Exception:
            state["answer"] = {"answer": "", "explain": "Failed to parse response"}
    return state