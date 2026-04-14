import logging
from dataclasses import dataclass, asdict
from typing import Dict, Optional
from mcp.server.fastmcp import FastMCP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class StoreData:
    """Domain model for store statistics."""
    user_cnt: int
    manager_cnt: int

class StoreManager:
    """
    Handles the business logic for managing store counts.
    This implementation uses an in-memory store; data will reset on server restart.
    """
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
        """
        Update a specific count for a store.
        Returns the updated StoreData or None if store/key is invalid.
        """
        store = self.get_store(store_name)
        if not store:
            logger.error(f"Store not found: {store_name}")
            return None

        if not hasattr(store, key):
            logger.error(f"Invalid metric key: {key}")
            return None

        # Update the attribute dynamically
        current_val = getattr(store, key)
        setattr(store, key, current_val + delta)
        
        logger.info(f"Updated {store_name} {key} by {delta}. New value: {getattr(store, key)}")
        return store

# Initialize FastMCP server
mcp = FastMCP("StoreCountServer")
store_manager = StoreManager()

@mcp.tool()
def get_store_count(store_name: str) -> str:
    """
    Get the current user and manager counts for a specific store.
    
    Args:
        store_name: The name of the store (e.g., 'STORE1').
    """
    store = store_manager.get_store(store_name)
    if not store:
        return f"Error: Store '{store_name}' not found."
    
    return f"Store {store_name.upper()} stats: {asdict(store)}"

@mcp.tool()
def update_store_count(store_name: str, metric: str, delta: int) -> str:
    """
    Update the count of a specific metric for a store.
    
    Args:
        store_name: The name of the store (e.g., 'STORE1').
        metric: The metric to update ('user_cnt' or 'manager_cnt').
        delta: The amount to add (positive) or subtract (negative).
    """
    updated_store = store_manager.update_count(store_name, metric, delta)
    if not updated_store:
        return f"Error: Failed to update {metric} for store {store_name}. Check if store and metric are valid."
    
    return f"Successfully updated {store_name.upper()} {metric}. New state: {asdict(updated_store)}"

if __name__ == "__main__":
    mcp.run()
