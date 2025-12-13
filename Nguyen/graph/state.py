from typing_extensions import TypedDict, Optional

class State(TypedDict):
    qid: str
    category: str
    message: str
    answer: dict
    rag_context: Optional[str]