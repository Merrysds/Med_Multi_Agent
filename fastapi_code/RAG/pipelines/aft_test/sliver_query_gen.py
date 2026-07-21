'''
对最后的材料的每一个chunk，生成对应的可能的query，调用LLM
'''

from services.llm import call_llm
import json
import asyncio

CHUNK_FILE = "RAG/datasets/brain/ZZ/final_all/final_chunk.json"
with open(CHUNK_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

def build_prompt(chunk_content):
    return [
        {
            "role": "user",
            "content": f"""
            你是一名严格的医学教育数据构造专家，负责为RAG系统生成高质量评测问题。

            ## 任务
            根据下面的“神经科教材内容”，生成 **2个高质量客观问题（fact-based QA）**。

            ---

            ## 生成要求（必须严格遵守）

            ### 1. 问题来源约束
            - 问题必须可以在原文中**直接找到答案**
            - 不得引入外部知识
            - 不得进行推理扩展到原文之外

            ---

            ### 2. 问题类型要求
            必须是以下类型之一：
            - 定义类（是什么）
            - 分类类（有哪些）
            - 机制类（如何发生，但必须原文明确描述）
            - 列举类（有哪些表现/特点）

            ---

            ### 3. 明确禁止
            - ❌ “本章主要讲述…”
            - ❌ “图中显示… / 表格说明…”
            - ❌ 无法从文本直接回答的问题
            - ❌ 需要跨段落推理才能回答的问题
            - ❌ 主观解释类问题

            ---

            ### 4. 输出要求（非常重要）
            - 只输出问题
            - 每行一个问题
            - 不要编号（不要1. 2.）
            - 不要解释
            - 不要多余字符
            - 如果无法生成任何合格问题，**直接输出NULL**

            ---

            ## 示例输出格式

            什么是XXX？
            XXX的主要临床表现有哪些？

            ---

            ## 教材内容
            {chunk_content}
            """
        }
    ]

async def generate_questions_for_chunks(chunks):
    results = []

    for idx, chunk in enumerate(chunks):
        content = chunk["content"].strip()
        block_idx_global_cut = chunk.get("block_idx_global_cut", "")

        # ====================== 内容过短直接跳=====================
        if not content or len(content) < 20:  # 小于20字符直接跳过
            print(f"[{idx+1}/{len(chunks)}] 内容过短，跳过")
            continue

        print(f"[{idx+1}/{len(chunks)}] 正在生成问题...")

        resp = await call_llm(build_prompt(content))
        try:
            if "choices" in resp:
                questions = resp["choices"][0]["message"]["content"].strip()
            elif "content" in resp:
                questions = resp["content"].strip()
            else:
                questions = str(resp)

            qs = [q.strip() for q in questions.split("\n") if q.strip() and q.strip() != "NULL"]
            print(qs)
            if len(qs) == 0 :
                continue

            for q in qs:
                results.append({
                    "query": q,
                    "answer": str(block_idx_global_cut)  # 存 block_idx 序号
                })

        except Exception as e:
            print(f"出错：{e}")
            continue

    return results

async def main():
    qa_dataset = await generate_questions_for_chunks(chunks)

    with open("RAG/datasets/brain/ZZ/test_data/test_dataset1.json", "w", encoding="utf-8") as f:
        json.dump(qa_dataset, f, ensure_ascii=False, indent=2)

    print("\n测评集生成完成：RAG/datasets/brain/ZZ/test_data/test_dataset1.json")

if __name__ == "__main__":
    asyncio.run(main())