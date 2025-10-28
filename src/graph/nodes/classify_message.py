from pydantic import BaseModel, Field
from typing_extensions import Literal

from src.graph.state import GraphState 

class MessageClassifier(BaseModel):
    message_type: Literal["emotional", "logical"] = Field(
        ...,
        description="Classify if the message requires an emotional (therapist) or logical response."
    )


def classify_message(state: GraphState, llm):
    last_message = state["messages"][-1]
    classifier_llm = llm.with_structured_output(MessageClassifier)

    result = classifier_llm.invoke([
        {
            "role": "system",
            "content": """Classify the user message as either:
            - 'emotional': if it asks for emotional support, therapy, deals with feelings, or personal problems
            - 'logical': if it asks for facts, information, logical analysis, or practical solutions
            ONLY respond with one of these two words, nothing else.
            """
        },
        {"role": "user", "content": last_message.content}
    ])
    return {"message_type": result.message_type}