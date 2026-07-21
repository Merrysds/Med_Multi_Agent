from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct

client = QdrantClient(host="qdrant", port=6333)

#collection存 4个维度
if not client.collection_exists("test"):
    client.create_collection(
        collection_name="test",
        vectors_config=VectorParams(
            size=4,
            distance=Distance.COSINE
        )
    )

# 2. 写入数据
client.upsert(
    collection_name="test",
    points=[
        PointStruct(
            id=1,
            vector=[0.1, 0.2, 0.3, 0.4],
            payload={"text": "hello qdrant"}
        )
    ]
)

#用余弦相似度直接找的最相近的文本
result = client.query_points(
    collection_name="test",
    query=[0.1, 0.2, 0.3, 0.4],
    limit=1
)
print(result)