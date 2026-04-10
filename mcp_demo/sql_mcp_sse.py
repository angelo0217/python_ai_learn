from typing import Any, List, Dict, Optional, Union
import asyncio
import json
import os
import aiosqlite
from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from mcp.server.sse import SseServerTransport
from starlette.requests import Request
from starlette.routing import Mount, Route
from starlette.responses import JSONResponse
from mcp.server import Server
import uvicorn
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

from core.config import settings
from core.logger import logger
from core.exceptions import DatabaseError

# 初始化 FastMCP 伺服器
mcp = FastMCP("sql_operator")

class SQLRepository:
    """封裝資料庫操作的 Repository 類別"""
    def __init__(self, db_path: str = settings.DB_PATH):
        self.db_path = db_path

    async def execute_query(
        self, query: str, params: Optional[Union[List, Dict]] = None
    ) -> List[Dict[str, Any]]:
        """
        執行 SQL 查詢並回傳結果。
        :param query: SQL 查詢字串
        :param params: 查詢參數
        :return: 結果列表
        :raises DatabaseError: 當資料庫操作失敗時拋出
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # 確保回傳結果為字典格式
                db.row_factory = aiosqlite.Row
                async with db.execute(query, params or []) as cursor:
                    rows = await cursor.fetchall()
                    # 處理 DML 操作 (INSERT, UPDATE, DELETE)
                    if cursor.description is None:
                        return [{"affected_rows": cursor.rowcount, "last_row_id": cursor.lastrowid}]
                    
                    return [dict(row) for row in rows]
        except aiosqlite.Error as e:
            logger.error(f"Database error executing query {query}: {e}")
            raise DatabaseError(f"SQL execution failed: {str(e)}")
        except Exception as e:
            logger.exception(f"Unexpected error during SQL execution: {e}")
            raise DatabaseError(f"An unexpected error occurred: {str(e)}")

# 實例化 Repository
repo = SQLRepository()

@mcp.tool()
async def execute_sql(query: str, params: str = "[]") -> str:
    """
    執行 SQL 查詢並回傳 JSON 結果。
    :param query: SQL 查詢字串
    :param params: JSON 格式的參數列表或字典
    """
    try:
        parsed_params = json.loads(params) if params else None
        results = await repo.execute_query(query, parsed_params)
        return json.dumps(results, ensure_ascii=False, indent=2)
    except json.JSONDecodeError:
        return "Error: Invalid JSON format for params."
    except DatabaseError as e:
        return f"Database Error: {str(e)}"
    except Exception as e:
        logger.exception(f"Unexpected error in execute_sql: {e}")
        return f"Unexpected Error: {str(e)}"

# --- SSE Server Setup ---
# 這裡保持原有的 Starlette 整合邏輯，但使用 logger 取代 print
async def main():
    # 建立 MCP 伺服器實例
    server = Server("sql_operator")
    
    # 這裡簡化原有的複雜路由，直接使用 FastMCP 的整合能力或維持原結構
    # 為了保持功能邏輯不變，我們保留 SSE 傳輸層
    sse = SseServerTransport("/messages", server)
    
    app = Starlette(
        routes=[
            Route("/sse", endpoint=sse.handle_sse),
            Route("/messages", endpoint=sse.handle_post_message, methods=["POST"]),
        ],
        middleware=[
            Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]),
        ]
    )
    
    logger.info(f"Starting SQL MCP SSE server on port {settings.MCP_PORT}...")
    uvicorn.run(app, host="0.0.0.0", port=settings.MCP_PORT)

if __name__ == "__main__":
    asyncio.run(main())
