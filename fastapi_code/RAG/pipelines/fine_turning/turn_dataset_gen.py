'''
构建用于微调的样本对
'''
import json
import asyncio
from typing import List, Dict
import ast
import os


ALL_DATA = "RAG/datasets/brain/ZZ/final_all/final_chunk.json"
TEST_DATA = "RAG/datasets/brain/ZZ/test_data/test_result_final1.json"
SAVE_DATA = "RAG/datasets/brain/ZZ/turn_data/turn_datasets.json"

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_turn_data(data, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

all_data = load_json(ALL_DATA)
test_data = load_json(TEST_DATA)
print(f"len(all):{len(all_data)}")


turn_data: list[dict] = []
for i,blocks in enumerate(test_data):
    item = {
        "query": "",
        "positive": [],
        "negtive": []  
    }
    item["query"] = blocks["query"]

    #pos
    pos_idx = blocks["reference_answer"]
    for idx1 in pos_idx:
        item["positive"].append(all_data[idx1]["content"])

    #neg
    neg_idx = blocks["rerank_block_idx_global_cut"][-10:]
    neg_idx_cut = list(set(neg_idx) - set(pos_idx))
    for idx2 in neg_idx_cut:
        item["negtive"].append(all_data[idx2]["content"])
    
    turn_data.append(item)

save_turn_data(turn_data, SAVE_DATA)

