class EmbeddingService:

    def __init__(self):
        self.model = SentenceTransformer(
            "models/bge_large"
        )

    def encode(self,text):
        return self.model.encode(
            text,
            normalize_embeddings=True
        )