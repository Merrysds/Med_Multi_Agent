# kg_graph.py
from neo4j import GraphDatabase
from typing import List, Dict, Any, Optional

class KGGraph:
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password123",
    ):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    # =========================
    # 基础执行
    # =========================
    def close(self):
        self.driver.close()

    def run(self, cypher: str, params: Dict = None):
        """执行任意Cypher"""
        with self.driver.session() as session:
            result = session.run(cypher, params or {})
            return [record.data() for record in result]

    # =========================
    # 创建节点
    # =========================
    def create_node(self, label: str, props: Dict):
        """
        通用创建节点
        """
        cypher = f"""
        MERGE (n:{label} {{id: $id}})
        SET n += $props
        RETURN n
        """
        return self.run(cypher, {"id": props.get("id"), "props": props})

    # =========================
    # 创建关系
    # =========================
    def create_relation(
        self,
        from_label: str,
        from_id: str,
        rel: str,
        to_label: str,
        to_id: str,
    ):
        cypher = f"""
        MATCH (a:{from_label} {{id: $from_id}})
        MATCH (b:{to_label} {{id: $to_id}})
        MERGE (a)-[r:{rel}]->(b)
        RETURN a, r, b
        """
        return self.run(
            cypher,
            {
                "from_id": from_id,
                "to_id": to_id,
            },
        )

    # =========================
    # 病人病程写入（核心）
    # =========================
    def add_patient_event(
        self,
        patient_id: str,
        event_type: str,
        event_value: str,
        timestamp: str,
        extra: Dict = None,
    ):
        """
        例如：
        symptom / diagnosis / treatment / exam
        """
        cypher = """
        MERGE (p:Patient {id: $pid})
        CREATE (e:Event {
            id: randomUUID(),
            type: $etype,
            value: $value,
            timestamp: $ts,
            extra: $extra
        })
        MERGE (p)-[:HAS_EVENT]->(e)
        RETURN p, e
        """

        return self.run(
            cypher,
            {
                "pid": patient_id,
                "etype": event_type,
                "value": event_value,
                "ts": timestamp,
                "extra": extra or {},
            },
        )

    # =========================
    # 查询病程（时间线）
    # =========================
    def get_patient_timeline(self, patient_id: str):
        cypher = """
        MATCH (p:Patient {id: $pid})-[:HAS_EVENT]->(e)
        RETURN e
        ORDER BY e.timestamp ASC
        """
        return self.run(cypher, {"pid": patient_id})

    # =========================
    # 查询某种事件
    # =========================
    def get_events_by_type(self, patient_id: str, event_type: str):
        cypher = """
        MATCH (p:Patient {id: $pid})-[:HAS_EVENT]->(e)
        WHERE e.type = $etype
        RETURN e
        ORDER BY e.timestamp ASC
        """
        return self.run(
            cypher,
            {
                "pid": patient_id,
                "etype": event_type,
            },
        )

    # =========================
    # 病人整体图谱
    # =========================
    def get_patient_graph(self, patient_id: str):
        cypher = """
        MATCH (p:Patient {id: $pid})-[r]->(n)
        RETURN p, r, n
        """
        return self.run(cypher, {"pid": patient_id})