import logging
from typing import Dict, Optional
from mcp.server.fastmcp import FastMCP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# In-memory mock store data
# Using a simple dictionary. Since this is a demo, we'll manage state directly.
STORE_DATA: Dict[str, Dict[str, int]] = {
    "STORE1": {"user_cnt": 18, "manager_cnt": 2},
    "STORE2": {"user_cnt": 20, "manager_cnt": 0},
}

mcp = FastMCP("StoreCountServer")

@mcp.tool()
def get_store_count(store_name: str) -> str:
    """
    Retrieve the user and manager counts for a specific store.
    
    Args:
        store_name: The name of the store (e.g., 'STORE1').
    """
    name = store_name.upper()
    data = STORE_DATA.get(name)
    
    if not data:
        return f"Store {name} not found."
    
    return f"Store {name}: Users = {data['user_cnt']}, Managers = {data['manager_cnt']}"

@mcp.tool()
def update_store_count(store_name: str, key: str, delta: int) -> str:
    """
    Update the count for a specific store and key.
    
    Args:
        store_name: The name of the store (e.g., 'STORE1').
        key: The count type to update ('user_cnt' or 'manager_cnt').
        delta: The amount to add (positive) or subtract (negative).
    """
    name = store_name.upper()
    key = key.lower()
    
    if name not in STORE_DATA:
        return f"Error: Store {name} not found."
    
    store = STORE_DATA[name]
    if key not in store:
        return f"Error: Invalid key '{key}'. Valid keys are 'user_cnt', 'manager_cnt'."
    
    store[key] += delta
    logger.info(f"Updated {name} {key} by {delta}. New value: {store[key]}")
    
    return f"Successfully updated {name} {key}. New count: {store[key]}"

if __name__ == "__main__":
    mcp.run()
