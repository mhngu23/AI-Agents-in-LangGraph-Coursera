from src.graph.state import GraphState

def router(state: GraphState):
    # print("ROUTER input:", state)

    message_type = state.get("message_type", "logical")
    if message_type == "emotional":
        state["next"] = "therapist"
    else:
        state["next"] = "logical"
    # print("ROUTER output:", state)
    return state