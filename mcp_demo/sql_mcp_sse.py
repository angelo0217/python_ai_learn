import logging
import os
import aiosqlite
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from mcp.server.fastmcp import FastMCP
from mcp_demo.mcp_utils import create_sse_app, run_sse_server

# Logging setup
logger = logging.getLogger(__name__)

# Data Models
class QueryResult(BaseModel):
    columns: List[str]
    rows: List[List[Any]]
    row_count: int

# Database Path Resolution
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../database.sqlite"))

mcp = FastMCP("sql_server")

async def execute_query(query: str, params: tuple = ()) -> QueryResult:
    """Helper to execute SQL and return a structured result."""
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                columns = [col[0] for col in cursor.description] if cursor.description else []
                return QueryResult(columns=columns, rows=rows, row_count=len(rows))
    except Exception as e:
        logger.error(f"SQL Execution Error: {e}")
        raise RuntimeError(f"Database error: {str(e)}")

@mcp.tool()
async def query_database(sql: str) -> str:
    """
    Execute a read-only SQL query on the database.
    Args:
        sql: The SQL SELECT statement to execute.
    """
    if not sql.strip().upper().startswith("SELECT"):
        return "Error: Only SELECT queries are allowed for this tool."
    
    result = await execute_query(sql)
    if not result.rows:
        return "No results found."
    
    output = [f"Columns: {', '.join(result.columns)}"]
    for row in result.rows:
        output.append(str(row))
    return "\n".join(output)

@mcp.tool()
async def list_tables() -> str:
    """List all tables in the database to understand the schema."""
    result = await execute_query("SELECT name FROM sqlite_master WHERE type='table';")
    return f"Tables: {', '.join([str(row[0]) for row in result.rows])}" if result.rows else "No tables found."

@mcp.tool()
async def describe_table(table_name: str) -> str:
    """Get the schema of a specific table."""
    result = await execute_query(f"PRAGMA table_info({table_name});")
    if not result.rows:
        return f"Table {table_name} not found."
    
    details = [f"{row[1]} ({row[2]})" for row in result.rows]
    return f"Schema for {table_name}: {', '.join(details)}"

if __name__ == "__main__":
    app, host, port = create_sse_app(mcp)
    run_sse_server(app, host, port)
