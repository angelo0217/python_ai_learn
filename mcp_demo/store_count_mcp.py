import logging
from typing import Dict, Optional
from mcp.server.fastmcp import FastMCP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("StoreCountServer")

# In-memory mock store data
# Using a dictionary to simulate a database
STORE_DATA: Dict[str, Dict[str, int]] = {
    "STORE1": {"user_cnt": 18, "manager_cnt": 2},
    "STORE2": {"user_cnt": 20, "manager_cnt": 0},
}

@mcp.tool()
def get_store_count(store_name: str) -> str:
    """
    Retrieve the current user and manager counts for a specific store.
    
    Args:
        store_name: The name of the store (e.g., 'STORE1', 'STORE2').
    """
    name_upper = store_name.upper()
    data = STORE_DATA.get(name_upper)
    
    if not data:
        return f"Error: Store '{store_name}' not found."
    
    return f"Store {name_upper} current counts: {data}"

@mcp.tool()
def update_store_count(store_name: str, key: str, delta: int) -> str:
    """
    Update the count for a specific metric in a store.
    
    Args:
        store_name: The name of the store (e.g., 'STORE1', 'STORE2').
        key: The metric to update ('user_cnt' or 'manager_cnt').
        delta: The amount to add (positive) or subtract (negative).
    """
    name_upper = store_name.upper()
    if name_upper not in STORE_DATA:
        return f"Error: Store '{store_name}' not found."
    
    store = STORE_DATA[name_upper]
    if key not in store:
        return f"Error: Invalid metric key '{key}'. Available keys: {list(store.keys())}"
    
    store[key] += delta
    logger.info(f"Updated {name_upper} {key} by {delta}. New value: {store[key]}")
    
    return f"Successfully updated {name_upper} {key}. New count: {store[key]}"

if __name__ == "__main__":
    mcp.run()
