import copy
import json
import logging
from typing import Any

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)


# In-memory mock store data (使用 deep copy 保護原始資料)
default_store = {
    "STORE1": {"user_cnt": 18, "manager_cnt": 2},
    "STORE2": {"user_cnt": 20, "manager_cnt": 0},
}


def _get_store_dict(store_name: str) -> dict[str, int] | None:
    """
    安全地取得 store 的字典副本。
    
    Args:
        store_name: Store 名稱。
    
    Returns:
        該 store 的字典副本，若不存在則返回 None。
    """
    return copy.deepcopy(default_store.get(store_name.upper(), None))


def _update_store(store_dict: dict[str, int], key: str, delta: int) -> None:
    """
    更新 store 字典中的計數。
    
    Args:
        store_dict: Store 字典（副本）。
        key: 計數鍵名 ('user_cnt' 或 'manager_cnt')。
        delta: 增量（正數增加，負數減少）。
    """
    store_dict[key] += delta


def _format_response(store_name: str, store_dict: dict[str, int]) -> str:
    """
    格式化回傳回應。
    
    Args:
        store_name: Store 名稱。
        store_dict: Store 字典。
    
    Returns:
        格式化後的回應字串。
    """
    return f"now {store_name} user {json.dumps(store_dict)}"


# Create MCP server
mcp = FastMCP("StoreManager")


# Tool: add user/manager
@mcp.tool()
def add_user(store_name: str, is_manager: bool) -> str:
    """
    新增使用者或經理到指定 store。
    
    Args:
        store_name: Store 名稱 (例如 "STORE1", "STORE2")。
        is_manager: True 為新增經理，False 為新增一般使用者。
    
    Returns:
        JSON 字串格式的更新後 count。
    
    Raises:
        RpcError: Store 名稱不存在時。
    """
    store_dict = _get_store_dict(store_name)
    if not store_dict:
        return "store not found."
    
    _update_store(store_dict, "manager_cnt", 1) if is_manager else _update_store(store_dict, "user_cnt", 1)
    return _format_response(store_name, store_dict)


# Tool: remove user/manager
@mcp.tool()
def user_leave(store_name: str, is_manager: bool) -> str:
    """
    移除使用者或經理。
    計數不會低於零。
    
    Args:
        store_name: Store 名稱 (例如 "STORE1", "STORE2")。
        is_manager: True 為移除經理，False 為移除一般使用者。
    
    Returns:
        JSON 字串格式的更新後 count。
    
    Raises:
        RpcError: Store 名稱不存在時。
    """
    store_dict = _get_store_dict(store_name)
    if not store_dict:
        return "store not found."
    
    _update_store(store_dict, "manager_cnt", -1) if is_manager else _update_store(store_dict, "user_cnt", -1)
    return _format_response(store_name, store_dict)


# Tool: get store info
@mcp.tool()
def get_store_info(store_name: str) -> str:
    """
    取得指定 store 的使用者與經理 count。
    
    Args:
        store_name: Store 名稱 (例如 "STORE1", "STORE2")。
    
    Returns:
        JSON 字串格式的 store info。
    
    Raises:
        RpcError: Store 名稱不存在時。
    """
    store_dict = _get_store_dict(store_name)
    if not store_dict:
        return "store not found."
    return f"{store_name} user info {json.dumps(store_dict)}"


# Resource: Greeting
@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """
    提供個人化問候訊息。
    
    Args:
        name: 被問候的人。
    
    Returns:
        個人化問候字串。
    """
    logger.info(f"Received greeting request for {name}")
    return f"Hello, {name}! This is the Store Manager Service."


if __name__ == "__main__":
    mcp.run(transport="stdio")
