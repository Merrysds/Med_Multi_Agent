# fastapi_code/skills/retrieval/rerank_skill.py
from skills.base import BaseSkill, BaseSkillInput, BaseSkillOutput
from sentence_transformers import CrossEncoder
from pydantic import Field

class RerankInput(BaseSkillInput):
    query: str = Field(description="用户查询问题")
    docs: list[dict] = Field(description="待重排的文档列表")
    top_n: int = Field(default=5, description="最终保留数量")

class RerankOutput(BaseSkillOutput):
    ranked_docs: list[dict] = Field(description="重排后的文档列表")

class RerankSkill(BaseSkill):
    """重排技能：对检索结果进行精排"""
    
    def __init__(self):
        # 加载 CrossEncoder 模型
        self.model = CrossEncoder(
            "models/jina-reranker",
            trust_remote_code=True
        )
    async def run_skill(self, input: RerankInput) -> RerankOutput:
        if not input.docs:
            return RerankOutput(ranked_docs=[])

        # 构造 (query, doc) 配对
        pairs = [[input.query, doc["content"]] for doc in input.docs]
        
        # 批量预测相关性分数
        scores = self.model.predict(pairs)
        
        # 将分数回填到文档中
        for doc, score in zip(input.docs, scores):
            doc["rerank_score"] = float(score)
            
        # 按分数降序排序并截取 Top N
        ranked = sorted(input.docs, key=lambda x: x["rerank_score"], reverse=True)
        
        return RerankOutput(ranked_docs=ranked[:input.top_n])