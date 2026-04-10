from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    # Redis Configuration
    REDIS_URL: str = "redis://localhost:6379"
    
    # AI Model Configuration
    OPENAI_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    XAI_API_KEY: Optional[str] = None
    
    # App Settings
    APP_NAME: str = "Python AI Learn"
    DEBUG: bool = True

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()
