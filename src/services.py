import torch
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from typing import Dict, Any, List, Optional, Tuple
import logging
from huggingface_hub import login
import os

logger = logging.getLogger(__name__)

class ModelService:
    def __init__(self):
        self._models = {}
        self._tokenizers = {}
        # login to huggingface
        print(os.getenv("HF_TOKEN"))
        login(token=os.getenv("HF_TOKEN"))

    def load_model(self, model_id: str, quantize: bool = False) -> None:
        """
        Load a model and tokenizer by ID and store them in memory.
        
        Args:
            model_id: The HuggingFace model ID
            quantize: Whether to use 8-bit quantization
        """
        # model_slug = model_id.replace("/", "-")
        if model_id in self._models:
            return  # Model already loaded
        
        logger.info(f"Loading model: {model_id} (quantize={quantize})")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Load model with optional quantization
        model_kwargs = {"device_map": "auto"}
        if quantize:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            model_kwargs["quantization_config"] = quantization_config
            
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id, 
            **model_kwargs
        )
        
        # Store model and tokenizer
        self._models[model_id] = model
        self._tokenizers[model_id] = tokenizer
        logger.info(f"Model {model_id} loaded successfully")
        
    def unload_model(self, model_slug: str) -> bool:
        """Unload a model from memory"""
        if model_slug in self._models:
            # Remove from memory
            del self._models[model_slug]
            del self._tokenizers[model_slug]
            return True
        return False
        
    def list_loaded_models(self) -> list:
        """List all loaded model IDs"""
        return list(self._models.keys())
        
    def _get_probabilities(self, model_slug: str, text: str, temperature: float = 1.0):
        """Internal method to get raw probabilities"""
        if model_slug not in self._models:
            raise ValueError(f"Model {model_slug} not loaded")
            
        model, tokenizer = self._models[model_slug], self._tokenizers[model_slug]
        
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            logits = model(**inputs).logits
        
        return softmax(logits / temperature, dim=-1)[0]
    
    def get_class_probabilities(self, model_id: str, text: str, temperature: float = 1.0) -> Tuple[List[float], List[str], Dict[str, float]]:
        """Get class probabilities for the input text along with their labels"""
        probs = self._get_probabilities(model_id, text, temperature)
        probs_list = probs.tolist()
        
        # Get labels from model config
        model = self._models[model_id]
        id2label = model.config.id2label
        
        # Create list of labels in the correct order
        labels = [id2label[i] for i in range(len(probs_list))]

        return probs_list, labels

# Create a singleton instance
model_service = ModelService()