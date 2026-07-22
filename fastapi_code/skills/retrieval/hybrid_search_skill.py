import jieba
import numpy as np
from rank_bm25 import BM25Okapi
from typing import List, Dict
from MCP.client import mcp_client

class HybridSearchSkill:
    """
    Skill:
    混合检索技能

    负责:
    1. BM25关键词检索
    2. 调用MCP vector_search工具
    3. RRF融合
    """
    def __init__(
        self,
        all_documents: List[Dict]
    ):
        self.all_documents = all_documents
        self._bm25_index = (
            self._build_bm25_index()
        )

    def _build_bm25_index(self):
        docs = [
            doc["content"]
            for doc in self.all_documents
        ]
        tokenized_docs = [
            jieba.lcut(doc)
            for doc in docs
        ]
        return BM25Okapi(
            tokenized_docs
        )

    def _bm25_search(
        self,
        query:str,
        top_k:int=20
    ):
        tokenized_query = jieba.lcut(query)
        scores = (
            self._bm25_index
            .get_scores(tokenized_query)
        )
        top_idx = np.argsort(scores)[::-1][:top_k]
        results=[]
        for idx in top_idx:
            doc = self.all_documents[idx].copy()
            doc["bm25_score"] = float(
                scores[idx]
            )
            doc["source_method"]="bm25"
            results.append(doc)
        return results

    async def _vector_search(
        self,
        query:str,
        top_k:int=20
    ):
        """
        调用 MCP Server
        """
        result = await mcp_client.call_tool(
            "vector_search",
            {
                "query":query,
                "top_k":top_k
            }
        )
        return result

    def _rrf_fusion(
        self,
        bm25_results,
        vector_results,
        top_k=20
    ):
        score_map={}
        k=60

        # BM25
        for rank,item in enumerate(
            bm25_results,
            1
        ):
            key = item["content"]
            if key not in score_map:
                score_map[key]={
                    **item,
                    "rrf_score":0
                }
            score_map[key]["rrf_score"] += (
                1/(rank+k)
            )

        # Vector
        for rank,item in enumerate(
            vector_results,
            1
        ):
            key=item["content"]
            if key not in score_map:
                score_map[key]={
                    **item,
                    "rrf_score":0
                }
            score_map[key]["rrf_score"] += (
                1/(rank+k)
            )
        return sorted(
            score_map.values(),
            key=lambda x:x["rrf_score"],
            reverse=True
        )[:top_k]

    async def run(
        self,
        query:str
    ):
        # 1 BM25
        bm25_results = self._bm25_search(
            query
        )
        # 2 MCP Vector Search
        vector_results = await self._vector_search(
            query
        )
        # 3 RRF
        final_results = self._rrf_fusion(
            bm25_results,
            vector_results
        )
        return final_results
