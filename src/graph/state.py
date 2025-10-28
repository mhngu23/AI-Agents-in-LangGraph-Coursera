from typing import Annotated, Literal, TypedDict

from langgraph.graph.message import add_messages

class GraphState(TypedDict):
    messages: Annotated[list, add_messages]
    message_type: str | None