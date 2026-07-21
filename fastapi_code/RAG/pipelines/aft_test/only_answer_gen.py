'''
测试：对测试集所有的query调用RAG,返回粗排和精排的结果
'''

import json
import asyncio
from graphs.rag_graph import rag_graph
import os
from services.llm import call_llm_Qwen_lan

FINAL_CHUNK_FILE =("RAG/datasets/brain/ZZ/final_all/final_chunk.json")  #最开始的所有文段

SAVE_LLM_RESULT = ("RAG/datasets/brain/ZZ/test_data/test_result_final6.json")   #LLM判定后的答案
REF_LLM_RESULT = ("RAG/datasets/brain/ZZ/test_data/test_result_final1.json")   #LLM判定后的答案


with open(REF_LLM_RESULT, "r", encoding="utf-8") as f:
        ref_data = json.load(f)

len(ref_data)

def save_test_data(data, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


async def run_test():

    with open(FINAL_CHUNK_FILE, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    print(len(all_data))

    results = []
    rerank_results = []
    # 用来存 LLM 最终结果
    llm_pred_results = []

    # test_data = [test_data[6]]
    for idx, item in enumerate(ref_data):

        query = item["query"]
        print(f"query:{query}")
        true_block_idx = item["true_block_idx"]

        # print(f"\n[{idx+1}/{len(test_data)}]")
        # print(query)

        state = await rag_graph.ainvoke({
            "query": query
        })

        results.append({
            "query": query,
            "query_idx": idx,
            "true_block_idx": true_block_idx,
            "rrf_block_idx_global_cut": state["rrf_block_idx_global_cut"],
            "rrf_score": state["rrf_score"]
        })
        print(state["rrf_block_idx_global_cut"])

        rerank_results.append({
            "query": query,
            "query_idx": idx,
            "true_block_idx": true_block_idx,
            "rrf_block_idx_global_cut": state["rrf_block_idx_global_cut"],
            "rerank_block_idx_global_cut": state["rerank_block_idx_global_cut"],
            "rerank_score": state["rerank_score"],
            "reference_answer": ref_data[idx]["reference_answer"]
        })
        print(state["rerank_block_idx_global_cut"])


    # for i, blocks in enumerate(rerank_results):
    #     print(i)
    #     query = blocks["query"]
    #     true_block_idx = blocks["true_block_idx"]
    #     # block_index_list = blocks["rerank_block_idx_global_cut"]
    #     L = len(all_data)
    #     block_index_list = list(range(0, L))

    #     # print(f"rerank_block_idx_global_cut:{block_index_list}")
    #     # 
    #     # print("all_data:")
    #     # print(len(all_data))
    #     # print("block_index_list:")
    #     # print(block_index_list)
    #     context_text = "\n\n".join([
    #         f"块[{all_data[b]['block_idx_global_cut']}]：{all_data[b]['content']}"
    #         for b in block_index_list
    #     ])

    #     print(query)
    #     # print(context_text)
    #     prompt = [
    #         {
    #             "role": "user",
    #             "content": f"""
    #             你是医学领域的RAG评测专家，负责从检索结果中筛选**能够直接回答用户问题的相关文本块**。

    #             ---

    #             ## 任务说明
    #             给定一个用户问题和若干检索块（每个块格式为：块编号 + 内容），请判断哪些块**包含可以直接用于回答该问题的关键信息**。

    #             ---

    #             ## 判断标准（必须满足）
    #             一个块被选中，必须满足至少一个条件：

    #             1. 能直接回答问题（包含明确事实/定义/机制/结论）
    #             2. 提供问题所需的关键医学信息（病因、机制、症状、分类、诊断等）
    #             3. 是回答该问题不可缺少的支持信息

    #             ---

    #             ## 重要限制
    #             - ❌ 不要选择仅提到关键词但无法回答问题的块
    #             - ❌ 不要选择泛泛背景描述
    #             - ❌ 不要选择与问题无关或弱相关内容
    #             - ❌ 最多只能选择 5 个块

    #             ---

    #             ## 输出要求（非常重要）
    #             - 只输出块编号列表
    #             - 按相关性排序（最相关的在前）
    #             - 不要解释
    #             - 不要输出任何多余字符
    #             - 如果没有任何相关块，输出空列表 []

    #             ---

    #             ## 输出格式示例
    #             [1, 3, 5]

    #             ---

    #             ## 用户问题
    #             {query}

    #             ---

    #             ## 检索块
    #             {context_text}
    #             """
    #         }
    #     ]

    #     resp = await call_llm_Qwen_lan(prompt)

    #     try:
    #         if "choices" in resp:
    #             res = resp["choices"][0]["message"]["content"].strip()
    #         else:
    #             res = resp.get("content", "").strip()

    #         import ast
    #         pred_list = ast.literal_eval(res)
            
    #         # 确保是列表 + 去重
    #         if isinstance(pred_list, list):
    #             pred_block_idx = list(set([int(x) for x in pred_list if str(x).isdigit()]))
    #         else:
    #             pred_block_idx = []

    #     except:
    #         pred_block_idx = []

    #     if len(pred_block_idx) == 0:
    #         print(f"跳过空结果：{query}")
    #         continue
        
    #     true_block_idx = ast.literal_eval(true_block_idx)
    #     true_block_idx = [true_block_idx]
    #     print(f"true_block_idx:{true_block_idx}")
    #     reference_answer = true_block_idx.copy()
    #     reference_answer.extend(pred_block_idx)
    #     print(f"reference_answer:{reference_answer}")
    #     reference_answer = list(set(reference_answer))#去重
    #     if len(reference_answer)>5:
    #         continue
    #     llm_pred_results.append({
    #         "query": query,
    #         "rrf_block_idx_global_cut": blocks["rrf_block_idx_global_cut"],
    #         "rerank_block_idx_global_cut": blocks["rerank_block_idx_global_cut"],
    #         "true_block_idx": true_block_idx,
    #         "pred_block_idx": pred_block_idx, # 直接存成列表
    #         "reference_answer": reference_answer
    #     })


    # 保存最终 LLM 结果
    save_test_data(rerank_results, SAVE_LLM_RESULT)
    print(f"LLM 预测完成！已保存到：{SAVE_LLM_RESULT}")
        

if __name__ == "__main__":
    asyncio.run(run_test())