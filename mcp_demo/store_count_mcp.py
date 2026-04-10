import logging
from dataclasses import dataclass, asdict
from typing import Dict, Optional
from mcp.server.fastmcp import FastMCP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class StoreData:
    """Represents the count of users and managers in a store."""
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

    def update_count(self, store_name: str, key: str, delta: int) -> Optional[StoreData]:
        """Update a specific count for a store."""
        store = self.get_store(store_name)
        if not store:
            logger.error(f"Store {store_name} not found.")
            return None
        
        if not hasattr(store, key):
            logger.error(f"Invalid attribute {key} for StoreData.")
            return None
        
        current_val = getattr(store, key)
        setattr(store, key, current_val + delta)
        return store

# Initialize FastMCP server
mcp = FastMCP("StoreCountServer")
store_manager = StoreManager()

@mcp.tool()
def get_store_count(store_name: str) -> str:
    """
    Get the current user and manager counts for a specific store.
    
    Args:
        store_name: The name of the store to query.
    """
    store = store_manager.get_store(store_name)
    if not store:
        return f"Error: Store '{store_name}' not found."
    
    return f"Store {store_name.upper()} counts: {asdict(store)}"

@mcp.tool()
def update_store_count(store_name: str, field: str, amount: int) -> str:
    """
    Update the count of users or managers for a specific store.
    
    Args:
        store_name: The name of the store to update.
        field: The field to update ('user_cnt' or 'manager_cnt').
        amount: The amount to add (positive) or subtract (negative).
    """
    updated_store = store_manager.update_count(store_name, field, amount)
    if not updated_store:
        return f"Error: Failed to update {field} for store '{store_name}'. Please check store name and field."
    
    return f"Successfully updated {field} for {store_name.upper()}. New state: {asdict(updated_store)}"

if __name__ == "__main__":
    mcp.run()
