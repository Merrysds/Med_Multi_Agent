import httpx
import json
import base64
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv  
load_dotenv()  
import os

VLLM_TEXT_URL = "http://vllm:8000/v1/chat/completions"
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

# 图片转 base64
def image_to_base64(image_path):
    with Image.open(image_path) as img:
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        base64_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/jpeg;base64,{base64_str}"

# async def call_llm(messages):
#     payload = {
#         "model": "/model",
#         "messages": messages,
#         "temperature": 0.7
#     }

#     async with httpx.AsyncClient(timeout=30) as client:
#         resp = await client.post(VLLM_URL, json=payload)
#         print(resp.status_code)
#         print(resp.text)
#         return resp.json()

#纯文本
async def call_llm_stream(messages):
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            "POST",
            VLLM_TEXT_URL,
            json={
                "model": "/model",
                "messages": messages,
                "stream": True
            }
        ) as resp:

            async for line in resp.aiter_lines():
                if not line:
                    continue

                if line.startswith("data: "):
                    data = line.replace("data: ", "")

                    if data == "[DONE]":
                        break

                    yield json.loads(data)

#纯文本
async def call_llm(messages):
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(VLLM_TEXT_URL, json={
            "model": "/model",
            "messages": messages,
            "temperature": 0.2
        })
        return resp.json()


async def call_llm_Qwen_vl(messages, image_path: str = None):
    # 如果有图片 → 构建多模态消息
    if image_path and len(messages) > 0:
        last_msg = messages[-1]
        text_content = last_msg["content"]
        image_base64 = image_to_base64(image_path)

        # 通义 VL 要求的格式
        messages[-1]["content"] = [
            {"type": "text", "text": text_content},
            {"type": "image_url", "image_url": {"url": image_base64}}
        ]
    
    url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json"
    }


    # 3. 调用 API（正确异步 httpx）
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            url=url,
            headers=headers,
            json={
                "model": "qwen-vl-plus",  # 看图选了这个
                "messages": messages,
                "temperature": 0.2
            }
        )

    result = resp.json()
    return result

async def call_llm_Qwen_lan(messages, image_path: str = None):
    # 如果有图片 → 构建多模态消息
    if image_path and len(messages) > 0:
        last_msg = messages[-1]
        text_content = last_msg["content"]
        image_base64 = image_to_base64(image_path)

        # 通义 VL 要求的格式
        messages[-1]["content"] = [
            {"type": "text", "text": text_content},
            {"type": "image_url", "image_url": {"url": image_base64}}
        ]
    
    url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json"
    }


    # 3. 调用 API（正确异步 httpx）
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            url=url,
            headers=headers,
            json={
                "model": "qwen3.6-plus",  # 看图选了这个
                "messages": messages,
                "temperature": 0.2
            }
        )

    result = resp.json()
    return result