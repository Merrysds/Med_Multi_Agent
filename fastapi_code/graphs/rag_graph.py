import json
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from graphs.rag_images_enrich_sub import build_image_subgraph
from graphs.rag_tables_enrich_sub import build_table_subgraph
import os
import shutil
from pathlib import Path
import re
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from qdrant_client.models import SearchParams, Filter, FieldCondition, MatchValue
from rank_bm25 import BM25Okapi
import jieba
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from services.llm import call_llm

client = QdrantClient(host="qdrant", port=6333)

FINAL_CHUNK_FILE = "RAG/datasets/brain/ZZ/final_all/"
FINAL_TABLE_FILE = "RAG/datasets/brain/ZZ/final_all/"
FINAL_IMAGE_FILE = "RAG/datasets/brain/ZZ/final_all/"
RAW_TABLE_FILE = "RAG/datasets/brain/ZZ/llm_enrich/table_meta.json"
RAW_IMAGE_FILE = "RAG/datasets/brain/ZZ/llm_enrich/image_meta.json"
RAW_ORIGIN_FILE = "RAG/datasets/brain/ZZ/raw/ZZ_raw_page_idx.json"
FINAL_DIR = "RAG/datasets/brain/ZZ/final_all"
FINAL_COPY_FILE = f"{FINAL_DIR}/ZZ_final_page_idx.json"
TEST_DATA = "RAG/datasets/brain/ZZ/rank_res/"

model_emb = SentenceTransformer('models/bge_large')
model_rerank = CrossEncoder(
    "models/jina-reranker",
    automodel_args={"torch_dtype": "auto"},
    trust_remote_code=True,
)

# reranker = FlagReranker(
#     'models/bge_rerank',
#     use_fp16=True,
#     use_fast=False
# )

query = "嗜睡是啥"

class RAGState(TypedDict):
    query: str
    doc_id: str
    tables: List[Dict]
    images: List[Dict]
    enriched_tables: List[Dict]
    enriched_images: List[Dict]
    embeddings: List[Dict]
    cache: bool
    final_blocks: List[Dict]
    ready_to_emb_blocks: List[Dict]
    rank_merge: List[Dict]
    #存粗排和精排结果的：
    rrf_block_idx_global_cut: List[int]
    rrf_score: List[float]
    rerank_block_idx_global_cut: List[int]
    rerank_score: List[float]
    #存最终选出来的结果
    rerank_full_list: List[Dict]
    #agent规划state:
    iteration: int
    max_iter: int
    need_refine: bool
    llm_response: str  

def clean(obj):
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    else:
        return obj
    
def init_copy_raw_file():
    Path(FINAL_DIR).mkdir(parents=True, exist_ok=True)
    if not Path(FINAL_COPY_FILE).exists():
        shutil.copy2(RAW_ORIGIN_FILE, FINAL_COPY_FILE)
        # print(f"原始文件已复制到：{FINAL_COPY_FILE}")

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

def save_data(data: List[Dict], filepath, name):
    full_path = os.path.join(filepath, name)
    os.makedirs(filepath, exist_ok=True) 
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(clean(data), f, ensure_ascii=False, indent=2)
    print(f"已保存增强结果到：{filepath+name}")

def update_block_in_final_file(block_index: int, new_block: dict):
    with open(FINAL_COPY_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    data[block_index] = new_block
    with open(FINAL_COPY_FILE, "w", encoding="utf-8") as f:
        json.dump(clean(data), f, ensure_ascii=False, indent=2)

async def run_table_graph(state: RAGState):
    # print(f"state_tables:{state['tables']}")
    #先读缓存
    if os.path.exists(FINAL_TABLE_FILE):
        full_path = os.path.join(FINAL_TABLE_FILE, "final_table.json")
        with open(full_path, "r", encoding="utf-8") as f:
            enriched_tables = json.load(f)
        print("表格增强缓存已加载，跳过重新生成")
        return {**state, "enriched_tables": enriched_tables}
    
    result = await table_graph.ainvoke({
        "tables": state["tables"]
    })

    enriched_tables = result["enriched_tables"]
    # print("\n")
    # print("!!!!!!")
    # print(enriched_tables)

    #直接更新掉原来的json
    for tb in enriched_tables:
        block_idx_global = tb['block_idx_global']
        llm_enrich_content = tb["llm_enrich"]

        new_block = {
            **tb,
            "llm_enrich": llm_enrich_content  
        }
        update_block_in_final_file(block_idx_global, new_block)

    #留一份单单表格的存档
    save_data(result["enriched_tables"], FINAL_TABLE_FILE, "final_table.json")
    return {
        **state,
        "enriched_tables": result["enriched_tables"]
    }
    
async def run_image_graph(state: RAGState):
    #先读缓存
    if os.path.exists(FINAL_IMAGE_FILE):
        full_path = os.path.join(FINAL_IMAGE_FILE, "final_image.json")
        with open(full_path, "r", encoding="utf-8") as f:
            enriched_images = json.load(f)
        print("图片增强缓存已加载，跳过重新生成")
        return {**state, "enriched_images": enriched_images}

    result = await image_graph.ainvoke({
        "images": state["images"]
    })
    enriched_images = result["enriched_images"]

    #直接更新掉原来的json
    for tb in enriched_images:
        block_idx_global = tb['block_idx_global']
        llm_enrich_content = tb["llm_enrich"]

        new_block = {
            **tb,
            "llm_enrich": llm_enrich_content  
        }
        update_block_in_final_file(block_idx_global, new_block)

    save_data(result["enriched_images"], FINAL_IMAGE_FILE, "final_image.json")
    return {
        **state,
        "enriched_images": result["enriched_images"]
    }

def should_skip_enrich(state: RAGState):

    #route
    if state["cache"]:
        return "chunk"
    else:
        return "run_table"

def split_text_with_full_context_overlap(merged_blocks, max_chunk=300):
    
    chunks = []
    for block_idx, block in enumerate(merged_blocks):
        text = block.get("content", "").strip()
        if not text:
            continue

        sentences = re.split(r'(?<=[。！？；\n])', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        current_group = []
        current_len = 0

        for s in sentences:
            current_group.append(s)

        if current_group:
            chunks.append(current_group)

    #overlap 前后一句话
    final_chunks = []
    total_blocks = len(chunks)
    # print(f"total_blocks:{total_blocks}\n")
    
    for i in range(total_blocks):
        current = chunks[i]
        # print(f"i:{i}\n")
        prev_sentence = []
        if i > 0:
            # print(f"chunks[i-1]:{chunks[i-1]}\n")
            prev_sentence = [chunks[i-1][-1]]  # 前一块最后一句
            # print(f"prev_sentence:{prev_sentence}")


        next_sentence = []
        if i < total_blocks - 1:
            next_sentence = [chunks[i+1][0]]  # 后一块第一句
            # print(f"next_sentence:{next_sentence}")

        # print(f"prev_sentence:{prev_sentence}")
        # print(f"current:{current}")
        # print(f"next_sentence:{next_sentence}")
        # print("\n")
        # 前 + 当前 + 后
        final_block = prev_sentence + current + next_sentence
        final_chunks.append(''.join(final_block))

    return final_chunks

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


async def emb(state: RAGState):
    # print("l(ready_to_emb_blocks):")
    # print(len(state["ready_to_emb_blocks"]))
    #目前先试了258条嵌入

    # sentences_1 = ["你是个大傻子"]
    # sentences_2 = ["你是大傻子吗？"]
    # model = FlagModel('models', 
    #                 query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
    #                 use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation
    # embeddings_1 = model.encode(sentences_1)
    # embeddings_2 = model.encode(sentences_2)
    # print(embeddings_1.shape)
    # similarity = embeddings_1 @ embeddings_2.T
    # print(f"similarity:{similarity}")

    if not client.collection_exists("med_no_tiao1"):
        client.create_collection(
            collection_name="med_no_tiao1",
            vectors_config=VectorParams(
                size=1024,
                distance=Distance.COSINE
            )
        )
    else:
    
        info = client.get_collection("med_no_tiao1")
        if info.points_count > 0:
            # print("之前已经嵌入好")
            return state  

    
    for i, block in enumerate(state["ready_to_emb_blocks"]):
        # print(f"i:{i}")
        content = block["content"]
        page = block.get("page", "")
        block_idx_innerpage = block.get("block_idx_innerpage", "")
        source = block.get("source", "")
        
        vector = model_emb.encode(content, normalize_embeddings=True)  #注意这个输出的是numpy数组
        payload = {
            "content": content,
            "page": page,
            "block_idx": block_idx_innerpage,
            "source": source
        }
        # print(vector)
        # print(payload)
        client.upsert(
            collection_name="med_no_tiao1",
            points=[
                PointStruct(
                    id=i,
                    vector=vector.tolist(),
                    payload=payload
                )
            ]
        )

    # print(client.count("med_no_tiao1"))
    # result = client.retrieve(
    #     collection_name="med_no_tiao1",
    #     ids=[0],
    #     with_vectors=True
    # )
    # print(result)
    
    return state

def bm25_search(query, docs, bm25, top_k=5):
    # 1. query分词
    tokenized_query = jieba.lcut(query)

    # 2. 计算BM25分数
    scores = bm25.get_scores(tokenized_query)

    # 3. 排序
    top_idx = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_idx:
        results.append({
            "index": idx,
            "score": float(scores[idx]),
            "content": docs[idx]
        })

    return results

def hybrid_fusion(bm25_results, hnsw_results, top_k=20):
    score_map = {}
    k = 60

    # BM25
    for rank, item in enumerate(bm25_results, 1):
        key = item["content"]
        
        if key not in score_map:
            score_map[key] = {
                **item,
                "score": 0.0
            }

        # score_map[key]["score"] += 1.0 / (rank + k)

    # HNSW
    for rank, point in enumerate(hnsw_results.points, 1):
        payload = point.payload
        key = payload["content"]

        if key not in score_map:
            score_map[key] = {
                **payload,            
                "score": 0.0
            }

        score_map[key]["score"] += 1.0 / (rank + k)

    merged = sorted(
        score_map.values(), #这里会有block_idx_global
        key=lambda x: x["score"],
        reverse=True
    )

    return merged[:top_k]


async def hybrid_rank(state: RAGState):

    query = state["query"]
    #BM25和Qdrant(HNSW)混合检索

    #关键词检索-----------------------------------------------------------------------------------
    #BM25中文分词（）-> BM25建立索引 -> query分词 ->  打分 -> 排序取topk
    blocks = state["ready_to_emb_blocks"]

    block_map = {
        block["content"]: block
        for block in state["ready_to_emb_blocks"]
    }
    
    docs = [block["content"] for block in blocks]

    tokenized_docs = [jieba.lcut(doc) for doc in docs]

    #倒排索引入内存
    bm25 = BM25Okapi(tokenized_docs)  #里面默认  k1:词频影响强度1.5  b:文档归一化长度 0.75

    #查询
    results_bm25 = bm25_search(query, docs, bm25, top_k=20)
    for r in results_bm25:
        meta = block_map[r["content"]]

        print(type(meta["page"]))
        print(type(meta["block_idx_innerpage"]))
        print(type(meta["block_idx_global_cut"]))
    
        r.update({
            "page": meta["page"],
            "block_idx_innerpage": meta["block_idx_innerpage"],
            "block_idx_global_cut": meta["block_idx_global_cut"],
            "source": meta["source"],
        })
    
    print("bm25_first_rank:")
    for r in results_bm25:
        print(r["score"])
        print(r["block_idx_global_cut"])
        print(r["content"])
        print("-----")

    #语义检索---------------------------------------------------------------------------------------
    instruction = "为这个句子生成表示以用于检索相关文章："
    query_vector = model_emb.encode(instruction + query, normalize_embeddings=True)

    results_hnsw = client.query_points(
        collection_name = "med_no_tiao1",
        query = query_vector.tolist(),
        limit = 20,
        score_threshold = 0.3,
        search_params = SearchParams(
            hnsw_ef = 128
        )
    )

    for point in results_hnsw.points:
        content = point.payload["content"]
        meta = block_map.get(content, {})

        print(type(meta.get("block_idx_global_cut", -1)))
        print(type(meta.get("block_idx_innerpage", -1)))
        point.payload.update({
            "block_idx_global_cut": meta.get("block_idx_global_cut", -1),
            "block_idx_innerpage": meta.get("block_idx_innerpage", -1),
        })


    print("len(results_hnsw):")
    print(len(results_hnsw.points))
    print("hnsw_first_rank:")
    for r in results_hnsw.points:
        print(r.score)
        print(r.payload["content"])
        print(r.payload["block_idx_global_cut"])
        print("-----")

    #RRF融合
    state["rank_merge"] = hybrid_fusion(results_bm25, results_hnsw, top_k=20)
    print("\nRRF融合之后：\n")
    for block in state["rank_merge"]:
        print(block)
        print("\n")
    save_data(state["rank_merge"], TEST_DATA, "aft_RRF.json")

    rrf_block_idx_global_cut = []
    rrf_score = []

    for block in state["rank_merge"]:
        rrf_block_idx_global_cut.append(block["block_idx_global_cut"])
        rrf_score.append(block["score"])

    state["rrf_block_idx_global_cut"] = rrf_block_idx_global_cut
    state["rrf_score"] = rrf_score
    return state

async def rerank(state: RAGState):

    query = state["query"]
    #精排
    blocks = state["rank_merge"]
    print("rank_merge::")
    print(state["rank_merge"])
    document = []

    for block in blocks:
        answer = block["content"]
        document.append(answer)
    
    sentence_pairs = [[query, doc] for doc in document]
    scores = model_rerank.predict(sentence_pairs, convert_to_tensor=True).tolist()
    
    rankings = model_rerank.rank(query, document, return_documents=True, convert_to_tensor=True)
    # print(f"Query: {query}")
    # for ranking in rankings:
    #     print(f"ID: {ranking['corpus_id']}, Score: {ranking['score']:.4f}, Text: {ranking['text']}")

    #tensor转普通字典
    rerank_block_idx_global_cut = []
    rerank_score = []

    save_rank_list = []
    for r in rankings:
        save_rank_list.append({
            **blocks[r["corpus_id"]],
            "in_ques_rank": r["corpus_id"],
            "score": float(r["score"]),  
            "answer": r["text"],
            
        })
        
        rerank_block_idx_global_cut.append(blocks[r["corpus_id"]]["block_idx_global_cut"])  #注意这里的下标，要用全局的下标，不要用内部排序的下标，不然每条就都是1-20
        rerank_score.append(float(r["score"]))

    save_data(save_rank_list, TEST_DATA, "aft_rerank.json")
    state["rerank_block_idx_global_cut"] = rerank_block_idx_global_cut
    state["rerank_score"] = rerank_score
    state["rerank_full_list"] = save_rank_list
    return state

async def inte_llm(state: RAGState):
    query = state["query"]
    top_all_refs = state["rerank_full_list"]

    # 1. 得分分层阈值
    HIGH_SCORE_THRESHOLD = 0.75   # 高可信度---完整原文
    MID_SCORE_THRESHOLD = 0.5     # 中等可信---精简摘要
    LOW_SCORE_THRESHOLD = 0.35    # 低可信度---仅保留关键片段

    high_priority = []
    mid_priority = []
    low_priority = []

    for block in top_all_refs:
        score = block["score"]
        if score >= HIGH_SCORE_THRESHOLD:
            high_priority.append(block)
        elif score >= MID_SCORE_THRESHOLD:
            mid_priority.append(block)
        else:
            low_priority.append(block)

    # 2. 分层拼接参考内容
    ref_sections = []
    ref_idx = 1

    # 高优：完整原文，权重最高
    for blk in high_priority[:3]:  # 高分最多取3条，避免过长
        ref_text = f"""【参考段落{ref_idx}｜高可信度 权重最高】
            书籍来源：{blk["source"]}
            页码：{blk["page"]}
            匹配得分：{round(blk["score"], 4)}
            完整原文：{blk["content"]}
        """
        ref_sections.append(ref_text)
        ref_idx += 1

    # 中优：精简截取，中等权重
    for blk in mid_priority[:2]:
        short_content = blk["content"][:350] + "……" if len(blk["content"]) > 350 else blk["content"]
        ref_text = f"""【参考段落{ref_idx}｜中等可信度 辅助参考】
            书籍来源：{blk["source"]}
            页码：{blk["page"]}
            匹配得分：{round(blk["score"], 4)}
            内容摘要：{short_content}
            """
        ref_sections.append(ref_text)
        ref_idx += 1

    # 低优：只留极短片段，仅作兜底，权重极低
    for blk in low_priority[:1]:
        tiny_content = blk["content"][:150] + "……" if len(blk["content"]) > 150 else blk["content"]
        ref_text = f"""【参考段落{ref_idx}｜低可信度 谨慎采信】
            书籍来源：{blk["source"]}
            页码：{blk["page"]}
            匹配得分：{round(blk["score"], 4)}
            少量片段：{tiny_content}
        """
        ref_sections.append(ref_text)
        ref_idx += 1

    all_reference = "\n=====\n".join(ref_sections)

    # 3. System Prompt 新增权重采信规则，指导LLM按分数区分重要性
    system_prompt = """
        你是一名专业神经内科医学顾问，仅依据提供的《神经病学》教材原文回答用户医学问题，严格遵守以下采信权重规则：
        # 采信优先级（严格遵守）
        1. 【高可信度｜权重最高】匹配得分≥0.75：优先完整采信，以此作为回答核心依据，重点展开说明；
        2. 【中等可信度｜辅助参考】匹配得分0.5~0.75：仅用来补充细节，不能作为核心结论；
        3. 【低可信度｜谨慎采信】匹配得分＜0.5：仅作兜底对照，存在冲突时直接忽略本条内容。

        # 硬性作答规则
        1. 所有回答内容必须完全来自下方给出的参考原文，严禁编造、拓展教材不存在的医学知识点；
        2. 若高可信度段落完全没有对应问题答案，直接回复：「现有教材资料中未查询到该问题相关内容，无法作答」，不要用低分内容强行猜测；
        3. 回答结构清晰，分点说明，专业名词通俗解释，符合临床科普阅读习惯；
        4. 回答中涉及的全部结论，末尾必须标注对应的【参考段落序号】，核心观点优先标注高分段落；
        5. 禁止输出和问题无关的冗余内容，优先提炼高可信度段落中的核心信息；
        6. 不夸大病情、不给出诊疗处方、不替代临床医师诊断，仅做教材知识科普。
        """

    user_prompt = f"""
        ### 分层参考教材原文（按匹配可信度划分权重）
        {all_reference}

        ### 用户提问
        {query}

        作答要求：
        1. 优先使用【高可信度】段落内容构建回答主体，详细展开；
        2. 中等可信度内容仅补充次要细节；低分内容仅作简单对照，冲突时舍弃；
        3. 所有结论标注对应参考段落编号，核心观点优先引用高分段落。
        """

    messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_prompt.strip()}
    ]

    llm_response = await call_llm(messages)
    print("llm_response:\n", llm_response)

    return {**state, "llm_response": llm_response}

#创子图 可能复用
table_graph = build_table_subgraph()
image_graph = build_image_subgraph()

rag_graph = StateGraph(RAGState)
rag_graph.add_node("material", material)
rag_graph.add_node("run_table", run_table_graph)
rag_graph.add_node("run_image", run_image_graph)
rag_graph.add_node("chunk", chunk)
rag_graph.add_node("emb", emb)
rag_graph.add_node("rank", hybrid_rank)
rag_graph.add_node("rerank", rerank)
rag_graph.add_node("inte_llm", inte_llm)
rag_graph.set_entry_point("material")

#分支
rag_graph.add_conditional_edges(
    "material",
    should_skip_enrich, #判断函数
    {
        "run_table": "run_table",   #函数返回的结果：下一个节点去哪里
        "chunk": "chunk"
    }
)

rag_graph.add_edge("run_table", "run_image")
rag_graph.add_edge("run_image", "chunk")
rag_graph.add_edge("chunk", "emb")
rag_graph.add_edge("emb", "rank")
rag_graph.add_edge("rank", "rerank")
rag_graph.add_edge("rerank", "inte_llm")
rag_graph.add_edge("inte_llm", END)

rag_graph = rag_graph.compile()

