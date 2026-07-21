from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional, Dict, Any
from graphs.kg_wrapper import KGGraph

kg = KGGraph(
    uri="bolt://neo4j:7687",
    user="neo4j",
    password="password123"
)


class KGState(TypedDict):
    query: str
    patient_id: Optional[str]
    intent: Optional[str]
    kg_result: Optional[Any]
    answer: Optional[str]


def router_node(state: KGState):
    query = state["query"]

    if "病史" in query or "病程" in query or "症状" in query:
        intent = "kg_query"
    elif "治疗" in query:
        intent = "kg_query"
    else:
        intent = "general"

    return {"intent": intent}


def kg_query_node(state: KGState):
    patient_id = state.get("patient_id", "001")

    result = kg.get_patient_timeline(patient_id)

    return {"kg_result": result}


def summarize_node(state: KGState):
    kg_data = state.get("kg_result")

    # 实际项目这里接 vLLM / OpenAI
    answer = f"患者病程记录：{kg_data}"

    return {"answer": answer}


def general_node(state: KGState):
    return {"answer": "这是一个普通问题，可以接LLM处理"}


def route_selector(state: KGState):
    return state["intent"]


workflow = StateGraph(KGState)

workflow.add_node("router", router_node)
workflow.add_node("kg_query", kg_query_node)
workflow.add_node("summarize", summarize_node)
workflow.add_node("general", general_node)

workflow.set_entry_point("router")

workflow.add_conditional_edges(
    "router",
    route_selector,
    {
        "kg_query": "kg_query",
        "general": "general",
    },
)

workflow.add_edge("kg_query", "summarize")
workflow.add_edge("summarize", END)
workflow.add_edge("general", END)

app = workflow.compile()