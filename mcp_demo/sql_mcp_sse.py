from typing import Any, List, Dict, Optional
import sqlite3
import logging
from mcp.server.fastmcp import FastMCP

# 設定日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sql_mcp")

# 初始化 FastMCP 伺服器
mcp = FastMCP("SQLServer")

DB_PATH = "mcp_demo.db"

def init_db():
    """初始化範例資料庫"""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT, email TEXT)")
        conn.execute("CREATE TABLE IF NOT EXISTS orders (id INTEGER PRIMARY KEY, user_id INTEGER, amount REAL, date TEXT)")
        # 插入一些範例數據
        conn.execute("INSERT OR IGNORE INTO users (id, name, email) VALUES (1, 'Alice', 'alice@example.com'), (2, 'Bob', 'bob@example.com')")
        conn.execute("INSERT OR IGNORE INTO orders (id, user_id, amount, date) VALUES (1, 1, 100.0, '2023-01-01'), (2, 1, 50.0, '2023-01-02'), (3, 2, 200.0, '2023-01-01')")
        conn.commit()

init_db()

@mcp.tool()
def query_database(sql: str) -> str:
    """
    執行 SQL 查詢並返回結果。
    
    Args:
        sql: 要執行的 SQL 查詢語句 (僅限 SELECT)。
    """
    if not sql.strip().upper().startswith("SELECT"):
        return "錯誤：僅允許執行 SELECT 查詢以確保安全性。"

    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()
            columns = [column[0] for column in cursor.description]
            
            if not rows:
                return "查詢完成，但沒有找到符合條件的資料。"
            
            # 格式化結果為表格形式
            header = " | ".join(columns)
            separator = "-" * len(header)
            body = "\n".join([" | ".join(map(str, row)) for row in rows])
            return f"{header}\n{separator}\n{body}"
    except sqlite3.Error as e:
        logger.error(f"SQL Error: {e}")
        return f"資料庫錯誤: {e}"
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return f"發生未預期錯誤: {e}"

@mcp.tool()
def list_tables() -> str:
    """列出資料庫中所有資料表名稱。"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            return "可用資料表: " + ", ".join([t[0] for t in tables])
    except Exception as e:
        return f"無法獲取資料表清單: {e}"

if __name__ == "__main__":
    mcp.run()
