from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any
from services.llm import call_llm_Qwen_vl
import os

class TableState(TypedDict):
    tables: List[Dict]
    enriched_tables: List[Dict]


async def enrich_table_node(state: TableState):
    enriched = []

    for t in state["tables"]:
        # print("ready to QWEN-vl!")
        # print(t)
        context = t["context"]
        image_path = t["content"]["image_source"]["path"]
        full_image_path = f"RAG/datasets/brain/ZZ/raw/{image_path}"

        print("图片路径:", full_image_path)
        print("文件是否存在:", os.path.exists(full_image_path))
        
        # 如果不存在，直接能看出来！
        if not os.path.exists(full_image_path):
            print("图片文件不存在！！！")

        messages = [
            {
                "role": "user",
                "content": f"""
                    请结合上下文和医学表格，请你根据表格的内容适当结合上下文（可能会出现上下文与表格无关的情况，此时忽略上下文），然后写一段关于表格内容的总述,注意包含细节内容，注意内容准确性：

                    上下文：{context}
                    输出JSON：
                    {{
                        "description": "",
                    }}
                    """
            }
        ]


        #call_llm_llava(messages, image_path: str = None):
        try:
            result = await call_llm_Qwen_vl(messages, full_image_path)
            
            content = result["choices"][0]["message"]["content"]
        except:
            content = "LLM 解析失败"

        enriched.append({
            **t,
            "llm_enrich": content
        })

    return {
        **state,
        "enriched_tables": enriched
    }


def build_table_subgraph():
    graph = StateGraph(TableState)

    graph.add_node("enrich_table", enrich_table_node)
    graph.set_entry_point("enrich_table")
    graph.add_edge("enrich_table", END)

    return graph.compile()