
from langgraph.graph import StateGraph

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from .state import GraphState
from src.graph.nodes.classify_message import classify_message
from src.graph.nodes.router import router
from src.graph.nodes.therapist_agent import therapist_agent
from src.graph.nodes.logical_agent import logical_agent



def build_graph(llm, token_callback=None):
    graph_builder = StateGraph(GraphState)

    def classify_wrapper(state):
        result = classify_message(state, llm)
        state.update(result)
        return state

    # def therapist_wrapper(state):
    #     result = therapist_agent(state, llm)
    #     state.update(result)
    #     return state

    # def logical_wrapper(state):
    #     result = logical_agent(state, llm)
    #     state.update(result)
    #     return state

    # Streaming wrapper for therapist agent
    def therapist_wrapper(state):
        assistant_message = {"role": "assistant", "content": ""}
        for token in therapist_agent(state, llm, stream=True):
            assistant_message["content"] += token.content
            state.update({"messages": [assistant_message]})
            if token_callback:
                token_callback(token, "therapist")
        return state

    # Streaming wrapper for logical agent
    def logical_wrapper(state):
        assistant_message = {"role": "assistant", "content": ""}
        for token in logical_agent(state, llm, stream=True):
            assistant_message["content"] += token.content
            state.update({"messages": [assistant_message]})
            if token_callback:
                token_callback(token, "logical")
        return state
    
    graph_builder.add_node("classifier", classify_wrapper)
    graph_builder.add_node("router", router)
    graph_builder.add_node("therapist", therapist_wrapper)
    graph_builder.add_node("logical", logical_wrapper)

    graph_builder.add_edge(START, "classifier")
    graph_builder.add_edge("classifier", "router")

    graph_builder.add_conditional_edges(
        "router",
        lambda state: state.get("next"),
        {"therapist": "therapist", "logical": "logical"}
    )

    graph_builder.add_edge("therapist", END)
    graph_builder.add_edge("logical", END)

    graph = graph_builder.compile()
    return graph