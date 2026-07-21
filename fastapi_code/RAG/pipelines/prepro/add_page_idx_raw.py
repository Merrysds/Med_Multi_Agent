import json
from pathlib import Path

RAW_PATH = "RAG/datasets/brain/ZZ/raw/ZZ_content_list_v2.json"
OUT_PATH = "RAG/datasets/brain/ZZ/raw/ZZ_raw_page_idx.json"

MAX_PAGES = 30

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

#按照json解析的格式先把json块给分出来
def flatten_blocks(data):
    """page -> block flatten"""
    blocks = []
    k = 0
    for page_id, page in enumerate(data[:MAX_PAGES]):
        for idx, block in enumerate(page):
            block["_page"] = page_id
            block["_idx"] = idx
            block["block_idx_global"] = k
            blocks.append(block)
            k += 1
    return blocks

def main():
    data = load_data()
    blocks = flatten_blocks(data)


    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(blocks, f, ensure_ascii=False, indent=2)

    print(f"Done: {len(blocks)} add page and idx")


if __name__ == "__main__":
    main()