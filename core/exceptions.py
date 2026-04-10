import logging
from typing import Optional

class BaseAppException(Exception):
    """Base class for all application exceptions."""
    def __init__(self, message: str, detail: Optional[any] = None):
        super().__init__(message)
        self.message = message
        self.detail = detail

class DatabaseError(BaseAppException):
    """Raised when a database operation fails."""
    pass

class ModelError(BaseAppException):
    """Raised when an AI model operation fails."""
    pass

class APIError(BaseAppException):
    """Raised when an external API call fails."""
    pass

class ConfigError(BaseAppException):
    """Raised when configuration is missing or invalid."""
    pass
