from typing import Any, List, Dict, Optional, Union
import asyncio
import logging
import os
import json
import aiosqlite
import uvicorn
from datetime import datetime
from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.routing import Mount, Route
from starlette.responses import JSONResponse
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from mcp.server.sse import SseServerTransport
from mcp.server import Server

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("sql_operator")

# Constants
DB_PATH = os.getenv("DB_PATH", "../database.sqlite")

class SQLOperator:
    """Handles SQL database operations with error handling and logging."""
    
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path

    async def execute_query(
        self, query: str, params: Optional[Union[List, Dict]] = None
    ) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Executes an SQL query and returns the results.
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(query, params or []) as cursor:
                    if query.strip().upper().startswith("SELECT") or query.strip().upper().startswith("PRAGMA"):
                        rows = await cursor.fetchall()
                        return [dict(row) for row in rows]
                    else:
                        await db.commit()
                        return {
                            "affected_rows": cursor.rowcount,
                            "last_row_id": cursor.lastrowid,
                            "status": "success"
                        }
        except aiosqlite.Error as e:
            logger.error(f"Database error executing query {query}: {e}")
            return {"error": f"Database error: {str(e)}"}
        except Exception as e:
            logger.exception(f"Unexpected error executing query {query}: {e}")
            return {"error": f"Unexpected error: {str(e)}"}

# Initialize FastMCP server
mcp = FastMCP("sql_operator")
sql_op = SQLOperator()

@mcp.tool()
async def query_database(query: str) -> str:
    """
    Execute a SQL query against the database.
    Args:
        query: The SQL query string.
    """
    result = await sql_op.execute_query(query)
    return json.dumps(result, indent=2, ensure_ascii=False)

@mcp.tool()
async def list_tables() -> str:
    """List all tables in the database."""
    result = await sql_op.execute_query("SELECT name FROM sqlite_master WHERE type='table';")
    return json.dumps(result, indent=2, ensure_ascii=False)

# SSE Server Setup
async def handle_sse(request: Request):
    async with SseServerTransport(request) as transport:
        # Note: In a real FastMCP scenario, the server is handled by the framework.
        # This part is kept for compatibility with the original SSE structure.
        await mcp.handle_request(transport)

app = Starlette(
    routes=[
        Route("/sse", endpoint=handle_sse),
        Mount("/mcp", app=mcp.app),
    ],
    middleware=[
        Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]),
    ]
)

if __name__ == "__main__":
    logger.info(f"Starting SQL MCP server on {DB_PATH}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
