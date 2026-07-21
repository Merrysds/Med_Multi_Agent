'''
测试：用于对粗排的结果检测指标
'''

import json
import asyncio
from typing import List, Dict
import ast

TEST_DATASET = "RAG/datasets/brain/ZZ/test_data/test_result_final_1.json"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

test_data = load_json(TEST_DATASET)

# rrf_data = rrf_data[:2]
# rrf_data = rrf_data[:2]


# Recall@K    能在topk中找到问题答案的问题/总的问题数
def recall_at_k(test_data, k=5):
    total = len(test_data)#总问题数量
    hit = 0
    print(f"total:{total}")

    key_map = {
        #问题：参考答案
        item["query"]: item["reference_answer"]
        for item in test_data
    }

    for i, item in enumerate(test_data):   #粗排的结果
        print(f"{i}:")
        query = item["query"]
        print(f"query:{query}")
        key_idx = key_map[query] #答案
        # key_idx = ast.literal_eval(key_idx) #字符串转成列表

        rrf_topk = item["rrf_block_idx_global_cut"][:k]  #粗排返回来的结果
        print(f"key_idx:{key_idx}")
        print(f"rrf_topk:{rrf_topk}")
        any_hit = any(int(idx) in map(int, rrf_topk) for idx in key_idx)
        if any_hit:
            hit += 1
        print(f"hit:{hit}")
    return hit / total

#MRR
def mrr(test_data):
    key_map = {
        item["query"]: item["reference_answer"]
        for item in test_data
    }

    scores = []
    for item in test_data:
        query = item["query"]
        key_idx = key_map[query]
        
        rrf = item["rrf_block_idx_global_cut"] 
        rank = None

        for idx, rrf_num in enumerate(rrf):
            if rrf_num in key_idx:  # 
                rank = idx + 1   # rank从1开始
                break
            
        # 3. 计算分数
        if rank is not None:
            scores.append(1.0 / rank)
        else:
            scores.append(0.0)

    # 平均 MRR
    return sum(scores) / len(scores) if len(scores) > 0 else 0.0


# -------------------------
# Evaluate
# -------------------------
def evaluate(test_data):

    r1 = recall_at_k(test_data, k=1)
    r2 = recall_at_k(test_data, k=2)
    r5 = recall_at_k(test_data, k=5)
    r10 = recall_at_k(test_data, k=10)
    r20 = recall_at_k(test_data, k=20)
    mrr_score = mrr(test_data)

    print(f"Recall@1:  {r1:.4f}")
    print(f"Recall@2:  {r2:.4f}")
    print(f"Recall@5:  {r5:.4f}")
    print(f"Recall@10: {r10:.4f}")
    print(f"Recall@20: {r20:.4f}")
    print(f"MRR:       {mrr_score:.4f}")

if __name__ == "__main__":
    evaluate(test_data)