from state import State
from langchain_vnpt.langchain_vnpt import LangChainVNPT
from langchain_core.messages import HumanMessage, SystemMessage
import pandas as pd
import json
from pathlib import Path

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
    state["final_answer"] = json.loads(response)
    return state

def reading_comprehension(state:State,
                          llm:LangChainVNPT,
                          prompt:str):
    response = llm.invoke([SystemMessage(content=prompt),
                          HumanMessage(content=state.get("message")).content
                        ]
                    )
    state["final_answer"] = json.loads(response)
    return state

def math_logic(state:State,
               llm:LangChainVNPT,
               prompt:str):
    response = llm.invoke([SystemMessage(content=prompt),
                          HumanMessage(content=state.get("message")).content
                        ]
                    )
    state["final_answer"] = json.loads(response)
    return state

def write_answer(state:State):
    qid = state.get("qid")
    answer = state.get("answer", {}).get("answer", "")
    
    df = pd.DataFrame([{"qid": qid, "answer": answer}])
    
    csv_path = Path(__file__).parent.parent / "results.csv"
    
    if csv_path.exists():
        df.to_csv(csv_path, mode='a', header=False, index=False, encoding='utf-8')
    else:
        df.to_csv(csv_path, mode='w', header=True, index=False, encoding='utf-8')
    
    return state