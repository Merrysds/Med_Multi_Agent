from mcp.server import Server
from mcp.server.stdio import stdio_server

from mcp.types import (
    Tool,
    TextContent
)

from MCP.tools.vector_db_tool import search_vectors


app = Server(
    "rag-mcp-server"
)

# ==========================
# 1. 暴露工具列表
# ==========================

@app.list_tools()
async def list_tools():

    return [
        Tool(
            name="vector_search",
            description="基于Qdrant向量数据库进行语义检索",
            inputSchema={
                "type": "object",
                "properties": {

                    "query":{
                        "type":"string",
                        "description":"用户查询"
                    },

                    "top_k":{
                        "type":"integer",
                        "description":"返回数量",
                        "default":20
                    }
                },

                "required":[
                    "query"
                ]
            }
        )
    ]



# ==========================
# 2. 工具调用入口
# ==========================


@app.call_tool()
async def call_tool(
    name:str,
    arguments:dict
):


    if name=="vector_search":


        query = arguments["query"]

        top_k = arguments.get(
            "top_k",
            20
        )


        result = search_vectors(
            query=query,
            top_k=top_k
        )


        docs=[]


        for point in result.points:

            docs.append(
                {
                    "content":
                        point.payload.get(
                            "content",
                            ""
                        ),

                    "page":
                        point.payload.get(
                            "page",
                            ""
                        ),

                    "source":
                        point.payload.get(
                            "source",
                            ""
                        ),

                    "score":
                        point.score
                }
            )


        return [
            TextContent(
                type="text",
                text=str(docs)
            )
        ]



    raise ValueError(
        f"Unknown tool:{name}"
    )

async def main():

    print(
        "MCP Server started: rag-mcp-server",
        flush=True
    )

    async with stdio_server() as streams:

        await app.run(
            streams[0],
            streams[1],
            app.create_initialization_options()
        )

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())