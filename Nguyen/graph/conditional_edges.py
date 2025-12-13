from .state import State
def semantic_router(state:State):
    return state.get("category")