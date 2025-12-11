from langgraph.graph import StateGraph, START, END
from state import State
from node import preprocessing, classifier, external_knowledge, reading_comprehension, math_logic, write_answer
from conditional_edges import preprocessing_router, semantic_router
from functools import partial
from langchain_vnpt.langchain_vnpt import LangChainVNPT
def create_graph(llm:LangChainVNPT,
                 tools:list,
                 external_knowledge_prompt:str,
                 reading_comprehension_prompt:str,
                 math_logic_prompt:str
                ):
    graph_builder = StateGraph(State)

    graph_builder.add_node("preprocessing", preprocessing)
    graph_builder.add_node("classifier", classifier)
    graph_builder.add_node("external_knowledge", partial(external_knowledge, 
                                                         llm.bind_tools(tools=tools), 
                                                         external_knowledge_prompt
                                                        )
                                                    )
    graph_builder.add_node("reading_comprehension", partial(reading_comprehension, 
                                                            llm, reading_comprehension_prompt
                                                        )
                                                    )
    graph_builder.add_node("math_logic", partial(math_logic, 
                                                 llm, 
                                                 math_logic_prompt
                                                )
                                            )
    graph_builder.add_node("write_answer", write_answer)

    graph_builder.add_edge(START, "preprocessing")
    
    graph_builder.add_conditional_edges("preprocessing",
                                        preprocessing_router,
                                        {}
                                        )
    
    graph_builder.add_conditional_edges("classifier",
                                        semantic_router,
                                        {}
                                        )
    
    graph_builder.add_edge("external_knowledge", "write_answer")
    graph_builder.add_edge("reading_comprehension", "write_answer")
    graph_builder.add_edge("math_logic", "write_answer")

    graph_builder.add_edge("write_answer", END)

    graph = graph_builder.compile()
    return graph