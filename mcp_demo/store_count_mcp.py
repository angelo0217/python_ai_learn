import copy
import json
import logging
from typing import Any, Optional, Dict

from mcp.server.fastmcp import FastMCP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# In-memory mock store data
DEFAULT_STORE: Dict[str, Dict[str, int]] = {
    "STORE1": {"user_cnt": 18, "manager_cnt": 2},
    "STORE2": {"user_cnt": 20, "manager_cnt": 0},
}

def get_store_data(store_name: str) -> Optional[Dict[str, int]]:
    """
    Safely retrieve a deep copy of the store data.
    
    Args:
        store_name: The name of the store.
        
    Returns:
        A copy of the store data if found, otherwise None.
    """
    data = DEFAULT_STORE.get(store_name.upper())
    return copy.deepcopy(data) if data else None

def update_store_count(store_dict: Dict[str, int], key: str, delta: int) -> None:
    """
    Update the count for a specific key in the store dictionary.
    
    Args:
        store_dict: The store data dictionary to update.
        key: The key to update ('user_cnt' or 'manager_cnt').
        delta: The amount to add (positive) or subtract (negative).
    """
    if key not in store_dict:
        logger.error(f"Invalid key: {key}. Available keys: {list(store_dict.keys())}")
        return
    store_dict[key] += delta

def format_store_response(store_name: str, store_dict: Dict[str, int]) -> str:
    """
    Format the store data into a JSON response string.
    
    Args:
        store_name: The name of the store.
        store_dict: The store data dictionary.
        
    Returns:
        A JSON string containing the store name and its data.
    """
    return json.dumps({
        "store_name": store_name, 
        "data": store_dict
    }, indent=2)

# Initialize FastMCP server
mcp = FastMCP("StoreCountServer")

@mcp.tool()
def get_store_count(store_name: str) -> str:
    """
    Retrieve the user and manager counts for a given store.
    """
    data = get_store_data(store_name)
    if data is None:
        return f"Store '{store_name}' not found."
    return format_store_response(store_name, data)

@mcp.tool()
def modify_store_count(store_name: str, key: str, delta: int) -> str:
    """
    Modify the count of users or managers for a given store.
    """
    data = get_store_data(store_name)
    if data is None:
        return f"Store '{store_name}' not found."
    
    update_store_count(data, key, delta)
    # Note: In a real scenario, you would save the modified 'data' back to a database.
    # For this mock, we just return the updated state of the copy.
    return format_store_response(store_name, data)

if __name__ == "__main__":
    mcp.run()
