from .state import State
from langchain_vnpt.langchain_vnpt import LangChainVNPT
from langchain_core.messages import HumanMessage, SystemMessage
import json

def preprocessing(state:State):
    return state

def classifier(state:State):
    return state

def external_knowledge(state:State, 
                       llm:LangChainVNPT,
                       prompt:str):
    response = llm.invoke([SystemMessage(content=prompt),
                          HumanMessage(content=state.get("message")).content
                        ]
                    )
    try:
        state["final_answer"] = json.loads(response)
    except json.JSONDecodeError as e:
        try:
            if '"answer"' in response:
                import re
                match = re.search(r'"answer"\s*:\s*"([A-Z])"', response)
                if match:
                    state["final_answer"] = {"answer": match.group(1), "explain": "Fallback due to JSON parse error"}
                else:
                    state["final_answer"] = {"answer": "", "explain": f"JSON parse error: {str(e)}"}
            else:
                state["final_answer"] = {"answer": "", "explain": f"JSON parse error: {str(e)}"}
        except Exception:
            state["final_answer"] = {"answer": "", "explain": "Failed to parse response"}
    return state

def reading_comprehension(state:State,
                          llm:LangChainVNPT,
                          prompt:str):
    response = llm.invoke([SystemMessage(content=prompt),
                          HumanMessage(content=state.get("message")).content
                        ]
                    )
    try:
        state["final_answer"] = json.loads(response)
    except json.JSONDecodeError as e:
        try:
            if '"answer"' in response:
                import re
                match = re.search(r'"answer"\s*:\s*"([A-Z])"', response)
                if match:
                    state["final_answer"] = {"answer": match.group(1), "explain": "Fallback due to JSON parse error"}
                else:
                    state["final_answer"] = {"answer": "", "explain": f"JSON parse error: {str(e)}"}
            else:
                state["final_answer"] = {"answer": "", "explain": f"JSON parse error: {str(e)}"}
        except Exception:
            state["final_answer"] = {"answer": "", "explain": "Failed to parse response"}
    return state

def math_logic(state:State,
               llm:LangChainVNPT,
               prompt:str):
    response = llm.invoke([SystemMessage(content=prompt),
                          HumanMessage(content=state.get("message")).content
                        ]
                    )
    try:
        state["final_answer"] = json.loads(response)
    except json.JSONDecodeError as e:
        try:
            if '"answer"' in response:
                import re
                match = re.search(r'"answer"\s*:\s*"([A-Z])"', response)
                if match:
                    state["final_answer"] = {"answer": match.group(1), "explain": "Fallback due to JSON parse error"}
                else:
                    state["final_answer"] = {"answer": "", "explain": f"JSON parse error: {str(e)}"}
            else:
                state["final_answer"] = {"answer": "", "explain": f"JSON parse error: {str(e)}"}
        except Exception:
            state["final_answer"] = {"answer": "", "explain": "Failed to parse response"}
    return state