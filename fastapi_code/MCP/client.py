from typing import Dict, Any


class MCPClient:
    """
    MCP Client封装

    负责：
    Skill -> MCP Server

    """

    def __init__(
        self,
        server_url:str="http://localhost:9000"
    ):

        self.server_url = server_url



    async def call_tool(
        self,
        tool_name:str,
        arguments:Dict[str,Any]
    ):

        """
        调用MCP Server中的Tool

        """

        print(
            f"[MCP Client] call tool: {tool_name}"
        )

        print(
            f"arguments:{arguments}"
        )


        # TODO:
        # 后面这里替换成真正MCP协议调用


        if tool_name=="vector_search":

            # 临时测试
            from MCP.tools.vector_db_tool import search_vectors


            result = search_vectors(
                query=arguments["query"],
                top_k=arguments.get(
                    "top_k",
                    20
                )
            )


            # 转换成普通dict
            docs=[]

            for point in result.points:

                docs.append(
                    {
                        **point.payload,
                        "score":point.score
                    }
                )


            return docs



        raise ValueError(
            f"Unknown tool:{tool_name}"
        )



# 单例

mcp_client = MCPClient()