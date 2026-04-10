from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """
    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # API Keys
    OPENAI_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    XAI_API_KEY: Optional[str] = None
    
    # Database
    DB_PATH: str = "database.sqlite"
    
    # App Settings
    APP_ENV: str = "development"
    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()
