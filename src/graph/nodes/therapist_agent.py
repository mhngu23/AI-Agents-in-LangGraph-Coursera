from src.graph.state import GraphState

def therapist_agent(state: GraphState, llm, stream=False):
    last_message = state["messages"][-1]

    messages = [
        {"role": "system",
         "content": """You are a compassionate therapist. Focus on the emotional aspects of the user's message.
                        Show empathy, validate their feelings, and help them process their emotions.
                        Ask thoughtful questions to help them explore their feelings more deeply.
                        Avoid giving logical solutions unless explicitly asked."""
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