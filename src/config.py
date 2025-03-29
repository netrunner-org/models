from pydantic_settings import BaseSettings
from typing import List, Dict, Any, Optional
import os

class Settings(BaseSettings):
    # API settings
    API_TITLE: str = "Model API"
    API_DESCRIPTION: str = "API for text classification models"
    API_VERSION: str = "0.1.0"
    
    # Model settings
    DEFAULT_MODELS: List[Dict[str, Any]] = [
        {
            "id": "meta-llama/Prompt-Guard-86M",
            "quantize": False
        }
    ]
    
    # Security settings
    API_KEY: Optional[str] = None

    HF_TOKEN: Optional[str] = None
    if HF_TOKEN is None:
        HF_TOKEN = os.getenv("HF_TOKEN")
    
    # Rate limiting settings
    RATE_LIMIT_CALLS: int = 100  # calls
    RATE_LIMIT_PERIOD: int = 60  # seconds
    
    # CORS settings
    CORS_ORIGINS: List[str] = ["*"]
    
    class Config:
        env_file = ".env"

settings = Settings()