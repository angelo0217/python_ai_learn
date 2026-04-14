import logging
from dataclasses import dataclass, asdict
from typing import Dict, Optional
from threading import Lock

from mcp.server.fastmcp import FastMCP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class StoreStats:
    """Store statistics model."""
    user_cnt: int
    manager_cnt: int

class StoreError(Exception):
    """Base exception for Store operations."""
    pass

class StoreNotFoundError(StoreError):
    """Raised when a store is not found."""
    pass

class InvalidKeyError(StoreError):
    """Raised when an invalid statistic key is provided."""
    pass

class StoreManager:
    """Manages in-memory store statistics with thread-safety."""
    
    def __init__(self, initial_data: Dict[str, Dict[str, int]]):
        self._lock = Lock()
        self._stores: Dict[str, StoreStats] = {
            k.upper(): StoreStats(**v) for k, v in initial_data.items()
        }

    def get_stats(self, store_name: str) -> StoreStats:
        """Retrieve statistics for a specific store."""
        name = store_name.upper()
        with self._lock:
            if name not in self._stores:
                raise StoreNotFoundError(f"Store '{store_name}' not found.")
            # Return a copy to prevent external mutation of the state
            stats = self._stores[name]
            return StoreStats(user_cnt=stats.user_cnt, manager_cnt=stats.manager_cnt)

    def update_count(self, store_name: str, field: str, delta: int) -> StoreStats:
        """Update a specific count field for a store."""
        name = store_name.upper()
        with self._lock:
            if name not in self._stores:
                raise StoreNotFoundError(f"Store '{store_name}' not found.")
            
            stats = self._stores[name]
            if not hasattr(stats, field):
                raise InvalidKeyError(f"Invalid field '{field}'. Valid fields are: user_cnt, manager_cnt")
            
            current_val = getattr(stats, field)
            setattr(stats, field, current_val + delta)
            
            return StoreStats(user_cnt=stats.user_cnt, manager_cnt=stats.manager_cnt)

# Initialize MCP Server
mcp = FastMCP("StoreCountServer")

# Initial Mock Data
DEFAULT_STORE_DATA = {
    "STORE1": {"user_cnt": 18, "manager_cnt": 2},
    "STORE2": {"user_cnt": 20, "manager_cnt": 0},
}

# Global manager instance
store_manager = StoreManager(DEFAULT_STORE_DATA)

@mcp.tool()
def get_store_count(store_name: str) -> str:
    """
    Retrieve the current user and manager counts for a specific store.
    
    Args:
        store_name: The name of the store (e.g., 'STORE1').
    """
    try:
        stats = store_manager.get_stats(store_name)
        return f"Store {store_name.upper()} stats: {asdict(stats)}"
    except StoreError as e:
        logger.error(f"Error retrieving store count: {e}")
        return f"Error: {str(e)}"

@mcp.tool()
def update_store_count(store_name: str, field: str, delta: int) -> str:
    """
    Update the count for a specific field in a store.
    
    Args:
        store_name: The name of the store (e.g., 'STORE1').
        field: The field to update ('user_cnt' or 'manager_cnt').
        delta: The amount to add (positive) or subtract (negative).
    """
    try:
        updated_stats = store_manager.update_count(store_name, field, delta)
        return f"Updated {store_name.upper()} {field}: {asdict(updated_stats)}"
    except StoreError as e:
        logger.error(f"Error updating store count: {e}")
        return f"Error: {str(e)}"

if __name__ == "__main__":
    mcp.run()
