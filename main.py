import os
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from anthropic import Anthropic

from src.graph.builder import build_graph

load_dotenv()  # Load environment variables from .env file

api_key = os.getenv("ANTHROPIC_API_KEY")


def run_chatbot():
    client = Anthropic(api_key=api_key)

    models = client.models.list()
    print(models)
    
    llm = init_chat_model(
        "anthropic:claude-3-haiku-20240307",
        api_key=api_key
    )
    
    graph = build_graph(llm)

    # Draw as PNG
    png_bytes = graph.get_graph().draw_png()

    # Save to file
    with open("output/graph.png", "wb") as f:
        f.write(png_bytes)
    # exit()

    state = {"messages": [], "message_type": None}

    while True:
        user_input = input("Message: ")
        if user_input == "exit":
            print("Bye")
            break

        state["messages"] = state.get("messages", []) + [
            {"role": "user", "content": user_input}
        ]

        state = graph.invoke(state)

        print(state["messages"])
        
        if state.get("messages") and len(state["messages"]) > 0:
            last_message = state["messages"][-1]
            print(f"Assistant: {last_message.content}")


if __name__ == "__main__":
    run_chatbot()