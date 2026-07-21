from fastapi import APIRouter, Request
import json
from fastapi.responses import StreamingResponse
import json
from services.memory import get_history, save_message
from services.llm import call_llm_stream
from core.prompt import build_messages
from graphs.chat_graph import chat_graph
from typing import AsyncGenerator

router = APIRouter()

@router.post("/chat")
async def chat(request: Request):
    body = await request.json()

    state = {
        "user_id" : body["user_id"],
        "session_id" : body["session_id"],
        "messages" : body.get("messages", []),
        "key" : f"{body['user_id']}:{body['session_id']}",
        "history" : [],
        "full_messages" : [],
        "assistant_messages" : "",
        "stream": AsyncGenerator
    }
    

    async def event_stream():
        async for event in chat_graph.astream_events(state, version = "v2"):
            print("EVENT:", event["event"])
            # call_model node结束
            if (
                event["event"] == "on_chain_end"
                and event["name"] == "call_model"
            ):
                stream = event["data"]["output"]["stream"]
                full_text = ""
                async for chunk in stream:
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        full_text += content
                        yield content
                
                #保存 assistant
                await save_message(state["key"], {
                    "role": "assistant",
                    "content": full_text
                })
        
    return StreamingResponse(event_stream(), media_type="text/plain")
    

@router.post("/chat/stream")
async def chat_stream(request: Request):
    body = await request.json()

    user_id = body["user_id"]
    session_id = body["session_id"]
    messages = body.get("messages", [])

    key = f"chat:{user_id}:{session_id}"
    #取历史
    history = await get_history(key)
    #给llm的
    full_messages = build_messages(history, messages)
    
    #存当前用户会话数据
    for m in messages:
        await save_message(key, m)

    async def event_stream():
        stream = call_llm_stream(full_messages)

        full_text = ""

        async for chunk in stream:
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content", "")

            if content:
                full_text += content

                yield f"data: {json.dumps({'content': content}, ensure_ascii=False)}\n\n"

        await save_message(key, {
            "role": "assistant",
            "content": full_text
        })

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


    



