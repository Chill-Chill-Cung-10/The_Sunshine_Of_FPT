from langgraph.graph import StateGraph, START, END
from .state import State
from .node import classifier, rag, external_knowledge, reading_comprehension, math_logic
from .conditional_edges import semantic_router
from functools import partial
from langchain_vnpt.langchain_vnpt import LangChainVNPT
def create_graph(llm:LangChainVNPT,
                 svm_model,
                 tools:list,
                 external_knowledge_prompt:str,
                 reading_comprehension_prompt:str,
                 math_logic_prompt:str
                ):
    graph_builder = StateGraph(State)

    graph_builder.add_node("classifier", partial(classifier, svm_model=svm_model))
    graph_builder.add_node("rag", rag)
    graph_builder.add_node("external_knowledge", partial(external_knowledge, 
                                                         llm=llm, 
                                                         prompt=external_knowledge_prompt
                                                        )
                                                    )
    graph_builder.add_node("reading_comprehension", partial(reading_comprehension, 
                                                            llm=llm, 
                                                            prompt=reading_comprehension_prompt
                                                        )
                                                    )
    graph_builder.add_node("math_logic", partial(math_logic, 
                                                 llm=llm, 
                                                 prompt=math_logic_prompt
                                                )
                                            )

    graph_builder.add_edge(START, "classifier")
    
    graph_builder.add_conditional_edges("classifier",
                                        semantic_router,
                                        {
                                            "external_knowledge": "rag",
                                            "reading_comprehension": "reading_comprehension",
                                            "math_logic": "math_logic"
                                        }
                                    )
    
    graph_builder.add_edge("rag", "external_knowledge")
    graph_builder.add_edge("external_knowledge", END)
    graph_builder.add_edge("reading_comprehension", END)
    graph_builder.add_edge("math_logic", END)

    graph = graph_builder.compile()
    return graph