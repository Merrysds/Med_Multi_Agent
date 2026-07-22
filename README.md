# **Med-Multi-Agents**
## 项目文档架构

DATA/（云端服务器盘名称）
├── docker/
│   ├── compose/
│   ├   ├── docker-compose.yml
├── fastapi_code/
│   ├── core/
│   │   ├── prompt.py
│   ├── graphs/
│   │   ├── chat_graph.py
│   │   ├── kg_workflow.py
│   │   ├── kg_wrapper.py
│   │   ├── rag_aft.py
│   │   ├── rag_graph.py
│   │   ├── rag_images_enrich_sub.py
│   │   ├── rag_tables_enrich_sub.py
│   │   └── test_rag_graph.py
│   ├── MCP/
│   │   ├── schemas/
│   │   ├── tools/
│   │   │   ├── __init__.py
│   │   │   ├── sql_db_tool.py
│   │   │   ├── vector_db_tool.py
│   │   │   └── web_search_tool.py
│   │   ├── __init__.py
│   │   ├── server.py
│   ├── models/
│   │   ├── bge_large/
│   │   ├── bge_large_finetuned/
│   │   ├── jina-reranker/
│   │   └── qwen_rerank/
│   ├── orchestrator/
│   │   ├── planner.py
│   │   ├── router.py
│   │   ├── state.py
│   │   └── supervisor.py
│   ├── RAG/
│   │   ├── datasets/
│   │   │   └── brain/
│   │   │       ├── bing1/
│   │   │       ├── NGA_brain/
│   │   │       ├── quanke_brain/
│   │   │       └── ZZ/
│   │   └── pipelines/
│   │       ├── aft_test/
│   │       │   ├── eval_rerank.py
│   │       │   ├── eval_RRF.py
│   │       │   ├── only_answer_gen.py
│   │       │   ├── sliver_answer_gen.py
│   │       │   └── sliver_query_gen.py
│   │       ├── fine_turning/
│   │       │   ├── turn_dataset_gen.py
│   │       │   ├── turn_plot_loss.py
│   │       │   └── turning.py
│   │       └── prepro/
│   │           ├── add_page_idx_raw.py
│   │           ├── enrich_image_meta.py
│   │           └── enrich_table_meta.py
│   ├── services/
│   ├   ├── llm.py
│   ├   └── memory.py
│   ├── skills/
│   │   ├── base.py
│   │   ├── eval/
│   │   │   └── relevance_eval_skill.py
│   │   │── generation/
│   │   ├── llm_eval/
│   │   │   ├── __init__.py
│   │   │   └── relevance_eval_skill.py
│   │   └── retrieval/
│   │       ├── basic_search_skill.py
│   │       ├── deep_search_skill.py
│   │       ├── hybrid_search_skill.py
│   │       └── rerank_skill.py
│   └── test/


HybridSearchSkill.py逻辑：
LangGraph

   |
   |
HybridSearchSkill.run()

   |
   |
   |------------------
   |                  |
   ↓                  ↓

 BM25              MCP Client

                    |
                    |
                    ↓

              MCP Server

                    |
                    |
                    ↓

             vector_db_tool

                    |
                    |
                    ↓

                 Qdrant
   |
   |
 RRF Fusion

   |
   |
 return docs
