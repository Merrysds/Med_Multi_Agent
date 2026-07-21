import json
from pathlib import Path

RAW_PATH = "RAG/datasets/brain/ZZ/raw/ZZ_raw_page_idx.json"
OUT_PATH = "RAG/datasets/brain/ZZ/llm_enrich/table_meta.json"

MAX_PAGES = 28

def load_data():
    with open(RAW_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

#这里面 
#  json的格式
#   [ 
#    #page:
#    [ 
#       {block}
#       {block} 
#       {block} 
#         ]
#          ]
# #

def get_context(blocks, i):
    prev_text = ""
    next_text = ""
    if i > 0:
        prev = blocks[i - 1]
        prev_text = extract_text(prev)
    if i < len(blocks) - 1:
        nxt = blocks[i + 1]
        next_text = extract_text(nxt)
    return prev_text, next_text


def extract_text(block):
    """把 paragraph/title 统一转文本"""
    print("see block:")
    print(block)
    t = block["type"]

    if t == "paragraph":
        return "".join([x["content"] for x in block["content"]["paragraph_content"]])
    if t == "title":
        return "".join([x["content"] for x in block["content"]["title_content"]])

    return ""


def build_table_meta(block, prev_text, next_text):
    # 这里先做 rule-based，后面你可以换LLM
    return {
        "type": "table",
        "description": "脑外科相关表格（需LLM增强）",
        "context": {
            "prev": prev_text[:300],
            "next": next_text[:300]
        },
        "content": block["content"],
        "bbox": block.get("bbox"),
        "_page": block.get("_page"),
        "_idx": block.get("_idx"),
        "block_idx_global": block.get("block_idx_global")
    }


def main():
    blocks = load_data()

    results = []

    for i, b in enumerate(blocks):
        #找tabble
        print(b)
        if b["type"] == "table":
            prev_text, next_text = get_context(blocks, i)
            meta = build_table_meta(b, prev_text, next_text)
            #加个下标，后续根据下标直接替换
            # meta["block_index"] = i
            results.append(meta)

    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Done: {len(results)} tables extracted")


if __name__ == "__main__":
    main()