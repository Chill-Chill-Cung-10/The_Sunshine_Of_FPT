from typing_extensions import TypedDict, Optional

class State(TypedDict):
    qid: str
    message: str
    answer: dict