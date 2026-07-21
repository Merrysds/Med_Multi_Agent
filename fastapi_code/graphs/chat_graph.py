from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from typing import AsyncGenerator
from services.memory import get_history, save_message
from services.llm import call_llm_stream
from core.prompt import build_messages

class ChatState(TypedDict):
    user_id: str
    session_id: str
    key: str
    messages: List[Dict[str, Any]]
    history: List[Dict[str, Any]]
    full_messages: List[Dict[str, Any]]
    assistant_messages: str
    stream: AsyncGenerator


async def load_memory(state: ChatState) -> ChatState:
    history = await get_history(state["key"])
    return {
        **state,
        "history": history
    }


async def build_prompt(state: ChatState) -> ChatState:
    full_messages = build_messages(
        state["history"],
        state["messages"]
    )
    return {
        **state,
        "full_messages": full_messages
    }


async def save_user_messages(state: ChatState) -> ChatState:
    for message in state["messages"]:
        await save_message(state["key"], message)

    return state


async def call_model(state: ChatState) -> ChatState:
    # full_text = ""

    # async for chunk in call_llm_stream(state["full_messages"]):
    #     delta = chunk.get("choices", [{}])[0].get("delta", {})
    #     content = delta.get("content", "")

    #     if content:
    #         full_text += content

    # return {
    #     **state,
    #     "assistant_message": full_text
    # }
    stream = call_llm_stream(state["full_messages"])
    return {
        **state,
        "stream": stream
    }


# async def save_assistant_message(state: ChatState) -> ChatState:
#     await save_message(state["key"], {
#         "role": "assistant",
#         "content": state["assistant_message"]
#     })

#     return state


workflow = StateGraph(ChatState)

workflow.add_node("load_memory", load_memory)
workflow.add_node("build_prompt", build_prompt)
workflow.add_node("save_user_messages", save_user_messages)
workflow.add_node("call_model", call_model)
# workflow.add_node("save_assistant_message", save_assistant_message)

workflow.set_entry_point("load_memory")

workflow.add_edge("load_memory", "build_prompt")
workflow.add_edge("build_prompt", "save_user_messages")
workflow.add_edge("save_user_messages", "call_model")
workflow.add_edge("call_model", END)
# workflow.add_edge("save_assistant_message", END)

chat_graph = workflow.compile()