async def chunk(state: RAGState):
    
    #先读缓存
    if os.path.exists(FINAL_CHUNK_FILE):
        full_path = os.path.join(FINAL_CHUNK_FILE, "final_chunk.json")
        with open(full_path, "r", encoding="utf-8") as f:
            ready_to_emb_blocks = json.load(f)
        state["ready_to_emb_blocks"] = ready_to_emb_blocks
        print("Chunk 缓存已加载，跳过重新分块")
        return state
    

    if state.get("cache"):
        blocks = state["final_blocks"]
    else:
        # 从 FINAL_COPY_FILE 读取
        with open(FINAL_COPY_FILE, "r", encoding="utf-8") as f:
            blocks = json.load(f)
    
    # 在这里做向量分块
    # print("进入 chunk 节点，开始生成向量...")
    # your chunk logic
    #按照层级先聚合
    # type = set()
    # for items in blocks:
    #     if items["type"] not in type:
    #         type.add(items["type"])

    # print(type)

    keep_types = {"title", "paragraph", "list", "table", "image"}
    
    merged = []
    current_text = ""  # 用来拼接连续的 title + paragraph

    for block in blocks:

        btype = block.get("type", "")
        
        if btype not in keep_types:
            continue
        
        #抽取metadata
        page = block.get("_page", "")          # 页码
        block_idx_innerpage = block.get("_idx", "")      # 块序号
        block_idx_global = block.get("block_idx_global", "")  # 全局块ID

        #按层级分 合并内容
        content = ""
        try:
            if btype == "title":
                parts = block["content"]["title_content"]
                content = "".join(p["content"] for p in parts)
            elif btype == "paragraph":
                parts = block["content"]["paragraph_content"]
                content = "".join(p["content"] for p in parts)
            elif btype == "list":
                items = block["content"]["list_items"]
                lines = []
                for item in items:
                    text_parts = item["item_content"]
                    line = "".join(p["content"] for p in text_parts)
                    lines.append(line)
                content = "\n".join(lines)
            elif btype == "table":
                content = block["llm_enrich"]
            elif btype == "image":
                content = block["llm_enrich"]
        except:
            content = f"[{btype} 内容解析失败]"

        block_data = {
            "type": btype,
            "content": content,
            "page": page,
            "block_idx_innerpage": block_idx_innerpage,
            "block_idx_global": block_idx_global,
            "source": "神经病学第8版"  
        }

        if btype == "paragraph":
            if merged and merged[-1]["type"] == "title":
                merged[-1]["content"] += "\n" + content
                merged[-1]["type"] = "title + paragraph"
            else:
                merged.append(block_data)
        elif btype == "title":
            merged.append(block_data)
        else:
            merged.append(block_data)
    
    ready_to_emb_blocks = []
    text_chunks = split_text_with_full_context_overlap(merged, max_chunk=300)

    # print(len(text_chunks))

    ready_to_emb_blocks = []
    for idx, chunk in enumerate(text_chunks):
        ready_to_emb_blocks.append({
            "type": "content",
            "content": chunk,
            "page": merged[idx]["page"],  
            "block_idx_innerpage": merged[idx]["block_idx_innerpage"],  
            "block_idx_global_cut": idx,  #注意这个地方重新编号，去掉跳过的那些从头编号
            "source": merged[idx]["source"],  
        })

    print("final_chunk长度：")
    print(len(ready_to_emb_blocks))

    save_data(ready_to_emb_blocks, FINAL_CHUNK_FILE, "final_chunk.json")
    state["ready_to_emb_blocks"] = ready_to_emb_blocks
    return state
