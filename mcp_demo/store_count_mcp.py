import json
import logging
from dataclasses import dataclass, asdict
from typing import Dict, Optional
from mcp.server.fastmcp import FastMCP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class StoreData:
    """Data model for store counts."""
    user_cnt: int
    manager_cnt: int

class StoreManager:
    """Manages the in-memory state of store counts."""
    def __init__(self):
        # Initial mock data
        self._stores: Dict[str, StoreData] = {
            "STORE1": StoreData(user_cnt=18, manager_cnt=2),
            "STORE2": StoreData(user_cnt=20, manager_cnt=0),
        }

    def get_store(self, store_name: str) -> Optional[StoreData]:
        """Retrieve store data by name (case-insensitive)."""
        return self._stores.get(store_name.upper())

    def update_count(self, store_name: str, key: str, delta: int) -> StoreData:
        """
        Update a specific count for a store.
        
        Args:
            store_name: Name of the store.
            key: The field to update ('user_cnt' or 'manager_cnt').
            delta: The amount to change (positive or negative).
            
        Raises:
            ValueError: If store is not found or key is invalid.
        """
        name_upper = store_name.upper()
        store = self._stores.get(name_upper)
        
        if store is None:
            raise ValueError(f"Store '{store_name}' not found.")
        
        if not hasattr(store, key):
            raise ValueError(f"Invalid count key: '{key}'. Valid keys are 'user_cnt', 'manager_cnt'.")
        
        # Update the attribute dynamically
        current_val = getattr(store, key)
        setattr(store, key, current_val + delta)
        
        return store

# Initialize FastMCP server
mcp = FastMCP("StoreCount")
store_manager = StoreManager()

@mcp.tool()
def get_store_count(store_name: str) -> str:
    """
    Get the current user and manager counts for a specific store.
    
    Args:
        store_name: The name of the store to query.
    """
    store = store_manager.get_store(store_name)
    if store is None:
        return f"Error: Store '{store_name}' not found."
    
    return json.dumps({
        "store_name": store_name.upper(),
        "data": asdict(store)
    }, indent=2)

@mcp.tool()
def update_store_count(store_name: str, key: str, delta: int) -> str:
    """
    Update the user or manager count for a specific store.
    
    Args:
        store_name: The name of the store to update.
        key: The count field to modify ('user_cnt' or 'manager_cnt').
        delta: The integer amount to add (use negative for subtraction).
    """
    try:
        updated_store = store_manager.update_count(store_name, key, delta)
        return json.dumps({
            "status": "success",
            "store_name": store_name.upper(),
            "updated_data": asdict(updated_store)
        }, indent=2)
    except ValueError as e:
        logger.error(f"Update failed: {e}")
        return f"Error: {str(e)}"

if __name__ == "__main__":
    # This allows running the server directly for testing
    mcp.run()
