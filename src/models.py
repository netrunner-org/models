from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Optional, Any
from src.services import model_service

# Request models
class TextClassificationRequest(BaseModel):
    model: str = Field(..., description="The model ID to use for classification")
    text: str = Field(..., min_length=1, description="The text to classify")
    temperature: float = Field(default=1.0, gt=0, description="Temperature for softmax")

    @field_validator('model')
    def validate_model_loaded(cls, v: str):
        if v not in model_service.list_loaded_models():
            raise ValueError('model doesn\'t exist or is not loaded')
        return v
    
    @field_validator('temperature')
    def temperature_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('temperature must be positive')
        return v

# Response models
class TextClassificationResponse(BaseModel):
    probabilities: List[float]
    labels: List[str]
    model_id: str

class HealthResponse(BaseModel):
    status: str
    loaded_models: List[str]

# Model registry models
class ModelInfo(BaseModel):
    id: str
    slug: str
    name: Optional[str] = None
    description: Optional[str] = None
    quantize: bool = False
    tags: List[str] = []
    metadata: Dict[str, Any] = {}