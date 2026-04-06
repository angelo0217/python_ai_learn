import logging
from typing import Any, Optional
from mcp.server.fastmcp import FastMCP

# 設定日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("store_count_mcp")

# 初始化 FastMCP 伺服器
mcp = FastMCP("StoreCountServer")

# 使用簡單的字典模擬資料庫
# 在實際生產環境中，這裡應該使用 Redis 或資料庫
storage = {}

@mcp.tool()
def update_count(key: str, increment: int = 1) -> str:
    """
    更新指定項目的計數。
    
    Args:
        key: 項目名稱 (例如: 'visits', 'clicks')
        increment: 增加的數值 (預設為 1)
    """
    current_value = storage.get(key, 0)
    new_value = current_value + increment
    storage[key] = new_value
    
    logger.info(f"Updated {key}: {current_value} -> {new_value}")
    return f"項目 '{key}' 已更新。目前總計: {new_value}"

@mcp.tool()
def get_count(key: str) -> str:
    """
    獲取指定項目的目前計數。
    
    Args:
        key: 項目名稱
    """
    value = storage.get(key, 0)
    return f"項目 '{key}' 的目前計數為: {value}"

@mcp.tool()
def reset_count(key: str) -> str:
    """
    將指定項目的計數重設為 0。
    
    Args:
        key: 項目名稱
    """
    storage[key] = 0
    return f"項目 '{key}' 已重設為 0。"

@mcp.tool()
def list_all_counts() -> str:
    """列出所有項目的計數。"""
    if not storage:
        return "目前沒有任何記錄。"
    
    results = [f"{k}: {v}" for k, v in storage.items()]
    return "目前所有計數:\n" + "\n".join(results)

if __name__ == "__main__":
    mcp.run()
