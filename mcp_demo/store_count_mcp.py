import copy
import json
import logging
from typing import Any, Optional, Dict
from dataclasses import dataclass, asdict

from mcp.server.fastmcp import FastMCP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class StoreData:
    user_cnt: int
    manager_cnt: int

class StoreManager:
    """
    Manages the in-memory store data with a thread-safe-like approach 
    using deep copies for data integrity.
    """
    def __init__(self, initial_data: Optional[Dict[str, Dict[str, int]]] = None):
        # Initialize with provided data or default mock data
        self._stores: Dict[str, StoreData] = {}
        
        data = initial_data or {
            "STORE1": {"user_cnt": 18, "manager_cnt": 2},
            "STORE2": {"user_cnt": 20, "manager_cnt": 0},
        }
        
        for name, values in data.items():
            self._stores[name.upper()] = StoreData(**values)

    def get_store(self, store_name: str) -> Optional[StoreData]:
        """Retrieve a copy of the store data."""
        data = self._stores.get(store_name.upper())
        return copy.deepcopy(data) if data else None

    def update_count(self, store_name: str, key: str, delta: int) -> Optional[StoreData]:
        """Update a specific count in the store and return the updated state."""
        store_name = store_name.upper()
        if store_name not in self._stores:
            logger.error(f"Store {store_name} not found.")
            return None
        
        store = self._stores[store_name]
        if not hasattr(store, key):
            logger.error(f"Invalid key: {key} for store {store_name}")
            return None
            
        current_val = getattr(store, key)
        setattr(store, key, current_val + delta)
        
        logger.info(f"Updated {store_name} {key} by {delta}. New value: {getattr(store, key)}")
        return copy.deepcopy(store)

    def format_response(self, store_name: str, store_data: StoreData) -> str:
        """Format the store data into a JSON string."""
        return json.dumps({
            "store_name": store_name.upper(), 
            "data": asdict(store_data)
        }, indent=2)

# Initialize FastMCP and StoreManager
mcp = FastMCP("StoreCountServer")
store_manager = StoreManager()

@mcp.tool()
def get_store_count(store_name: str) -> str:
    """
    Get the current user and manager counts for a specific store.
    
    Args:
        store_name: The name of the store (e.g., 'STORE1').
    """
    data = store_manager.get_store(store_name)
    if not data:
        return f"Error: Store '{store_name}' not found."
    
    return store_manager.format_response(store_name, data)

@mcp.tool()
def update_store_count(store_name: str, key: str, delta: int) -> str:
    """
    Update the count for a specific metric in a store.
    
    Args:
        store_name: The name of the store.
        key: The metric to update ('user_cnt' or 'manager_cnt').
        delta: The amount to change (positive to increase, negative to decrease).
    """
    updated_data = store_manager.update_count(store_name, key, delta)
    if not updated_data:
        return f"Error: Failed to update store '{store_name}' with key '{key}'."
    
    return store_manager.format_response(store_name, updated_data)

if __name__ == "__main__":
    mcp.run()
