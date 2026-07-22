from langgraph.graph import StateGraph,END

from skills.retrieval.chunk_skill import ChunkSkill
from skills.retrieval.hybrid_search_skill import HybridSearchSkill
from skills.retrieval.rerank_skill import RerankSkill
from skills.generation.answer_generate_skill import AnswerGenerateSkill


class RAGState(TypedDict):

    query:str

    chunks:list

    rank_merge:list

    rerank_docs:list

    answer:str



async def chunk_node(state):

    result = await chunk_skill.run_skill(
        state
    )

    return result



async def retrieve_node(state):

    result = await hybrid_skill.run_skill(
        state
    )

    return result



async def rerank_node(state):

    result = await rerank_skill.run_skill(
        state
    )

    return result



async def generate_node(state):

    result = await answer_skill.run_skill(
        state
    )

    return result



builder=StateGraph(RAGState)


builder.add_node(
"chunk",
chunk_node
)


builder.add_node(
"retrieve",
retrieve_node
)


builder.add_node(
"rerank",
rerank_node
)


builder.add_node(
"generate",
generate_node
)



builder.add_edge(
"chunk",
"retrieve"
)

builder.add_edge(
"retrieve",
"rerank"
)


builder.add_edge(
"rerank",
"generate"
)


builder.add_edge(
"generate",
END
)


rag_graph=builder.compile()