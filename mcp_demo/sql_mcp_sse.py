from typing import Any, List, Dict, Optional, Union
import asyncio
import json
import os
import sqlite3
import uvicorn
import aiosqlite
from datetime import datetime

from mcp.server.fastmcp import FastMCP
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

# --- Configuration ---
DB_PATH = os.getenv("DB_PATH", "../database.sqlite")
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))

# Initialize FastMCP server
mcp = FastMCP("sql_operator")

async def execute_query(
    query: str, params: Optional[Union[List, Dict]] = None
) -> List[Dict[str, Any]]:
    """
    Executes an SQL query and returns the results.
    Supports SELECT, PRAGMA, INSERT, UPDATE, DELETE, and other DDL/DML statements.

    Args:
        query: The SQL query string to execute.
        params: Optional parameters for the SQL query.
    
    Returns:
        A list of dictionaries representing the rows or operation metadata.
    """
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            # Set row factory to return dictionaries
            db.row_factory = aiosqlite.Row
            
            async with db.execute(query, params or []) as cursor:
                if query.strip().upper().startswith("SELECT") or query.strip().upper().startswith("PRAGMA"):
                    rows = await cursor.fetchall()
                    return [dict(row) for row in rows]
                else:
                    await db.commit()
                    return [{"affected_rows": cursor.rowcount, "last_row_id": cursor.lastrowid}]
    except Exception as e:
        return [{"error": str(e)}]

@mcp.tool()
async def query_database(sql: str) -> str:
    """
    Execute a read-only SQL query to retrieve data from the database.
    
    Args:
        sql: The SQL SELECT query to execute.
    """
    if not sql.strip().upper().startswith("SELECT"):
        return "Error: Only SELECT queries are allowed for this tool."
    
    results = await execute_query(sql)
    return json.dumps(results, indent=2, ensure_ascii=False)

@mcp.tool()
async def modify_database(sql: str) -> str:
    """
    Execute a DML SQL query to modify data (INSERT, UPDATE, DELETE).
    
    Args:
        sql: The SQL modification query to execute.
    """
    results = await execute_query(sql)
    return json.dumps(results, indent=2, ensure_ascii=False)

# --- SSE Integration ---
async def sse_endpoint(request: Request):
    """
    SSE endpoint to handle MCP server connections.
    """
    async with SseServerTransport("/messages") as transport:
        # This is a simplified representation of how SseServerTransport 
        # would be integrated with the FastMCP server logic.
        # In a real scenario, the transport would be linked to the server's request handler.
        await transport.connect()
        # Note: FastMCP usually handles the server loop; 
        # manual SSE integration requires linking the server instance to the transport.
        return JSONResponse({"status": "connected"})

routes = [
    Route("/sse", endpoint=sse_endpoint),
]

middleware = [
    Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]),
]

app = Starlette(routes=routes, middleware=middleware)

if __name__ == "__main__":
    # Run the server
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
