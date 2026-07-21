
import asyncio
import json
from graphs.rag_graph import rag_graph   

async def test():
    result = await rag_graph.ainvoke({
        "query":"嗜睡是啥",
        "doc_id": "test_001",
        "tables": [],
        "images": [],
        "enriched_tables": [],
        "enriched_images": [],
        "embeddings": []
    })
    # print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    asyncio.run(test())