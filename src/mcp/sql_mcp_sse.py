from typing import Any, List, Dict, Optional, Union
import asyncio
from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from mcp.server.sse import SseServerTransport
from starlette.requests import Request
from starlette.routing import Mount, Route
from starlette.responses import JSONResponse
from mcp.server import Server
import uvicorn
import sqlite3
import aiosqlite
import json
import os
from datetime import datetime
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

# 初始化 FastMCP 伺服器，命名為 "sql_operator"
mcp = FastMCP("sql_operator")

# 常數
DB_PATH = "../database.sqlite"  # 資料庫路徑


async def execute_query(
    query: str, params: Optional[Union[List, Dict]] = None
) -> List[Dict[str, Any]]:
    """
    Executes an SQL query and returns the results.
    Supports SELECT, PRAGMA, INSERT, UPDATE, DELETE, and other DDL/DML statements.

    :param query: The SQL query string to execute.
    :param params: Optional parameters for the SQL query (e.g., for prepared statements).
                   Can be a list for positional parameters or a dictionary for named parameters.
    :return: A list of dictionaries, where each dictionary represents a row.
             For DML operations (INSERT, UPDATE, DELETE), returns info like affected_rows and last_row_id.
             Returns a dictionary with an 'error' key if an exception occurs.
    """
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(query, params or [])

            if query.strip().upper().startswith(("SELECT", "PRAGMA")):
                rows = await cursor.fetchall()
                result = [dict(row) for row in rows]
                return result
            else:
                await db.commit()
                return [{"affected_rows": cursor.rowcount, "last_row_id": cursor.lastrowid}]
    except aiosqlite.Error as e:
        print(f"SQL 錯誤：{e}")
        return [{"error": str(e)}]
    except Exception as e:
        print(f"發生未預期的錯誤：{e}")
        return [{"error": str(e)}]


def init_database() -> None:
    """
    Initializes the SQLite database.
    Creates the 'users' table if it does not exist and inserts some test data.
    This function should be called once at application startup.
    """
    if not os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # 建立一個測試資料表
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE,
            age INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        )

        # 插入一些測試資料
        test_data = [
            ("張三", "zhang@example.com", 30),
            ("李四", "li@example.com", 25),
            ("王五", "wang@example.com", 35),
        ]
        cursor.executemany(
            "INSERT OR IGNORE INTO users (name, email, age) VALUES (?, ?, ?)", test_data
        )

        conn.commit()
        conn.close()
        print("資料庫已初始化")


def format_result(result: List[Dict[str, Any]]) -> str:
    """
    Formats the SQL query result into a human-readable string, typically a formatted table.
    Handles empty results, error messages, and DML operation summaries.

    :param result: A list of dictionaries representing the query's rows or an error/summary dictionary.
    :return: A string representation of the query result.
    """
    if not result:
        return "查詢完成，沒有返回資料。"

    if "error" in result[0]:
        return f"錯誤：{result[0]['error']}"

    if "affected_rows" in result[0]:
        return f"執行成功！影響的行數：{result[0]['affected_rows']}，最後插入的ID：{result[0]['last_row_id']}"

    # 將結果格式化為表格形式的字串
    headers = result[0].keys()
    rows = [[str(row.get(header, "")) for header in headers] for row in result]

    # 計算每列的最大寬度
    widths = [
        max(len(header), max([len(row[i]) for row in rows])) for i, header in enumerate(headers)
    ]

    # 創建分隔線
    separator = "+" + "+".join("-" * (width + 2) for width in widths) + "+"

    # 創建表頭
    header_str = "|" + "|".join(f" {h:<{widths[i]}} " for i, h in enumerate(headers)) + "|"

    # 創建資料行
    data_rows = []
    for row in rows:
        data_rows.append(
            "|" + "|".join(f" {cell:<{widths[i]}} " for i, cell in enumerate(row)) + "|"
        )

    # 組合表格
    table = [separator, header_str, separator] + data_rows + [separator]

    return f"查詢結果 ({len(result)} 筆資料)：\n" + "\n".join(table)


@mcp.tool()
async def execute_sql(query: str) -> str:
    """
    Executes a direct SQL query against the database.
    This tool is versatile for performing various database operations like SELECT, INSERT, UPDATE, DELETE, and DDL.
    The result will be formatted as a human-readable string, including table data for SELECTs or
    affected row counts for DML operations.

    :param query: The full SQL query string to be executed (e.g., "SELECT * FROM users WHERE age > 30;").
    :return: A formatted string containing the query results, or an error message if the query fails.
    :raises RpcError: If a database error or unexpected error occurs during query execution.
    """
    result = await execute_query(query)
    return format_result(result)


@mcp.tool()
async def get_table_structure(table_name: str) -> str:
    """
    Retrieves the structure (schema) of a specified table, including column names, types, and constraints.
    This is useful for understanding the columns available in a table.

    :param table_name: The name of the database table (e.g., "users").
    :return: A formatted string representing the table's schema in a tabular format, or an error if the table is not found.
    :raises RpcError: If a database error occurs (e.g., table not found).
    """
    query = f"PRAGMA table_info({table_name})"
    result = await execute_query(query)
    return format_result(result)


@mcp.tool()
async def list_tables() -> str:
    """
    Lists all tables present in the current database.
    This tool is helpful for discovering available tables before performing operations on them.

    :return: A formatted string listing all table names in a tabular format.
    :raises RpcError: If a database error occurs.
    """
    query = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    result = await execute_query(query)
    return format_result(result)


@mcp.tool()
async def insert_data(table_name: str, data_json: str) -> str:
    """
    Inserts a new record into the specified table.
    The data to be inserted must be provided as a JSON string.

    :param table_name: The name of the table to insert data into (e.g., "users").
    :param data_json: A JSON string representing the data to insert.
                      Example: '{"name": "Alice", "email": "alice@example.com", "age": 28}'.
                      Keys must match column names in the table.
    :return: A formatted string indicating the success of the insertion, including affected rows and last inserted ID,
             or an error message if the JSON is invalid or insertion fails.
    :raises RpcError: If a database error occurs or JSON is malformed.
    """
    try:
        data = json.loads(data_json)
        columns = ", ".join(data.keys())
        placeholders = ", ".join(["?"] * len(data))
        values = list(data.values())

        query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
        result = await execute_query(query, values)
        return format_result(result)
    except json.JSONDecodeError:
        return "錯誤：提供的 JSON 格式不正確"
    except Exception as e:
        return f"錯誤：{e}"


@mcp.tool()
async def update_data(table_name: str, data_json: str, condition: str) -> str:
    """
    Updates existing records in the specified table that match a given condition.
    The data to be updated must be provided as a JSON string.

    :param table_name: The name of the table to update (e.g., "users").
    :param data_json: A JSON string representing the column-value pairs to update.
                      Example: '{"name": "New Name", "age": 31}'.
                      Keys must match column names.
    :param condition: The SQL WHERE clause string (e.g., "id = 1", "age < 30 AND name = 'Bob'").
                      Do NOT include the "WHERE" keyword itself.
    :return: A formatted string indicating the success of the update, including affected rows,
             or an error message if the JSON is invalid or update fails.
    :raises RpcError: If a database error occurs or JSON is malformed.
    """
    try:
        data = json.loads(data_json)
        set_clause = ", ".join([f"{key} = ?" for key in data.keys()])
        values = list(data.values())

        query = f"UPDATE {table_name} SET {set_clause} WHERE {condition}"
        result = await execute_query(query, values)
        return format_result(result)
    except json.JSONDecodeError:
        return "錯誤：提供的 JSON 格式不正確"
    except Exception as e:
        return f"錯誤：{e}"


@mcp.tool()
async def delete_data(table_name: str, condition: str) -> str:
    """
    Deletes records from the specified table that match a given condition.
    Use with extreme caution when providing an empty condition, as it will delete all data.

    :param table_name: The name of the table to delete from (e.g., "users").
    :param condition: The SQL WHERE clause string (e.g., "id = 1", "age < 20").
                      Do NOT include the "WHERE" keyword itself.
                      If an empty string is provided, ALL data in the table will be deleted.
    :return: A formatted string indicating the success of the deletion, including affected rows,
             or an error message if deletion fails.
    :raises RpcError: If a database error occurs.
    """
    query = f"DELETE FROM {table_name}"
    if condition.strip():
        query += f" WHERE {condition}"

    result = await execute_query(query)
    return format_result(result)


@mcp.resource("tables://{table_name}/schema")
async def table_schema(table_name: str) -> JSONResponse:
    """
    Retrieves the schema (column information) of a specified database table as a JSON array.
    This resource is designed for programmatic access to table structure.

    :param table_name: The name of the table whose schema is to be retrieved.
    :return: A Starlette JSONResponse containing a list of dictionaries, where each dictionary
             describes a column (e.g., {"cid": 0, "name": "id", "type": "INTEGER", ...}).
             Returns an error JSON if the table is not found or a database error occurs.
    """
    query = f"PRAGMA table_info({table_name})"
    result = await execute_query(query)
    return json.dumps(result)


@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting"""
    return f"Hello, {name}!"


@mcp.prompt()
def table_structure_prompt(operation: str) -> str:
    """
    Provides a personalized greeting message from the SQL operator service.

    :param name: The name of the person to greet.
    :return: A personalized greeting string.
    """
    if operation == "get_table_structure":
        return f"you can get table structure <UNK>"
    else:
        return "Invalid operation. Please choose exist operation"


def create_starlette_app(mcp_server: Server, *, debug: bool = False) -> Starlette:
    """
    Generates a prompt string based on the requested operation.
    This prompt can guide an LLM on how to interact with specific tools.

    :param operation: The name of the operation or tool (e.g., "get_table_structure").
    :return: A descriptive string related to the operation, or an "Invalid operation" message.
    """
    sse = SseServerTransport("/messages/")

    async def handle_sse(request: Request) -> None:
        async with sse.connect_sse(
            request.scope,
            request.receive,
            request._send,  # noqa: SLF001
        ) as (read_stream, write_stream):
            await mcp_server.run(
                read_stream,
                write_stream,
                mcp_server.create_initialization_options(),
            )

    middleware = [
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    ]

    routes = [
        Route("/sse", endpoint=handle_sse),
        Mount("/messages/", app=sse.handle_post_message),
    ]

    return Starlette(
        debug=debug,
        routes=routes,
        middleware=middleware,
    )


if __name__ == "__main__":
    mcp_server = mcp._mcp_server

    import argparse

    parser = argparse.ArgumentParser(description="運行基於 SSE 的 MCP SQL 伺服器")
    parser.add_argument("--host", default="0.0.0.0", help="綁定的 Host")
    parser.add_argument("--port", type=int, default=8082, help="監聽的 Port")
    parser.add_argument("--db", default=DB_PATH, help="資料庫路徑")
    args = parser.parse_args()

    # 設定資料庫路徑
    DB_PATH = args.db

    # 初始化資料庫
    init_database()

    print(f"SQL MCP 伺服器啟動中，資料庫路徑：{DB_PATH}")
    print(f"SSE 訪問地址：http://{args.host}:{args.port}/sse")
    print(f"資料表 Schema 資源位於：http://{args.host}:{args.port}/tables/<table_name>/schema")
    print(f"所有資料表列表資源位於：http://{args.host}:{args.port}/tables")

    # 綁定 MCP 資源到 Starlette 應用程式
    starlette_app = create_starlette_app(mcp_server, debug=True)
    # mcp.mount_to_app(starlette_app)

    uvicorn.run(starlette_app, host=args.host, port=args.port)
