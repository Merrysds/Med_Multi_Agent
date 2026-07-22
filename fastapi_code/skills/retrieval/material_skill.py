def load_cache() -> List[Dict] | None:
    
    #直接看有没有LLM增强的数据，有的话直接调用
    if os.path.exists(FINAL_COPY_FILE):
        with open(FINAL_COPY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return None
    #llm_enrich就是中间层放一下增强出来的表格和图片
async def material(state: RAGState):
    cached = load_cache()
    if cached is not None:
        state["cache"] = True
        state["final_blocks"] = cached   # 加个字段存全文块
        return state  # 
    
    init_copy_raw_file()
    #第一次拼一下
    with open(RAW_TABLE_FILE) as f:
        tables = json.load(f)
        # print("tables_check:")
        # print(tables)
    with open(RAW_IMAGE_FILE) as f:
        images = json.load(f)
    state["cache"] = False

    # tables = tables[:1]
    return {
        **state,
        "tables": tables,
        "images": images
    }
