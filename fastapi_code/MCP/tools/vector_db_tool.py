# fastapi_code/mcp/tools/vector_db_tool.py
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct, SearchParams
from sentence_transformers import SentenceTransformer
from services.embedding import EmbeddingService
import os

# 没配
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "med_no_tiao1")

# 初始化客户端和模型
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
model_emb = SentenceTransformer('models/bge_large')

def create_collection_if_not_exists():
    """确保集合存在"""
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
        )

def search_vectors(query: str, top_k: int = 20, score_threshold: float = 0.3):
    """
    MCP Tool: 语义向量检索
    这是最基础的原子能力，只负责调用 Qdrant API。
    """
    create_collection_if_not_exists()
    query_vector = model_emb.encode(query, normalize_embeddings=True)
    
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector.tolist(),
        limit=top_k,
        score_threshold=score_threshold,
        search_params=SearchParams(hnsw_ef=128)
    )
    # 直接返回 Qdrant 的原始结果，不做任何业务处理
    return results

def upsert_vectors(points: list[PointStruct]):
    """
    MCP Tool: 向量写入/更新
    """
    create_collection_if_not_exists()
    client.upsert(collection_name=COLLECTION_NAME, points=points)