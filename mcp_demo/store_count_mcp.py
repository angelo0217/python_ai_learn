import copy
import json
import logging
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

# In-memory mock store data (使用 deep copy 保護原始資料)
default_store = {
    "STORE1": {"user_cnt": 18, "manager_cnt": 2},
    "STORE2": {"user_cnt": 20, "manager_cnt": 0},
}

def get_store_dict(store_name: str) -> Optional[dict[str, int]]:
    """
    安全地取得 store 的字典副本。  
    
    Args:
        store_name: Store 名稱。  
    
    Returns:
        該 store 的字典副本，若不存在則返回 None。
    """
    return copy.deepcopy(default_store.get(store_name.upper(), None))


def update_store(store_dict: dict[str, int], key: str, delta: int) -> None:
    """
    更新 store 字典中的計數。  
    
    Args:
        store_dict: Store 字典（副本）。
        key: 計數鍵名 ('user_cnt' 或 'manager_cnt')。
        delta: 增量（正數增加，負數減少）。
    """
    if key not in store_dict:
        logger.error(f"Invalid key: {key}")
        return
    store_dict[key] += delta


def format_response(store_name: str, store_dict: dict[str, int]) -> str:
    """
    格式化回傳回應。  
    
    Args:
        store_name: Store 名稱。
        store_dict: Store 字典。
    """
    # 假設這裡會返回一些格式化的字符串
    return json.dumps({"store_name": store_name, "data": store_dict})