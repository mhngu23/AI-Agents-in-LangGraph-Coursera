
from src.graph.state import GraphState

def logical_agent(state: GraphState, llm, stream=False):
    """A logical agent that provides fact-based responses."""
    last_message = state["messages"][-1]

    messages = [
        {"role": "system",
         "content": """You are a purely logical assistant. Focus only on facts and information.
            Provide clear, concise answers based on logic and evidence.
            Do not address emotions or provide emotional support.
            Be direct and straightforward in your responses."""
         },
        {
            "role": "user",
            "content": last_message.content
        }
    ]
    if stream:
        for token in llm.stream(messages):
            yield token
    else:
        reply = llm.invoke(messages)
        return {"messages": [{"role": "assistant", "content": reply.content}]}