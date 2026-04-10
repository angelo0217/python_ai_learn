import os
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelMerger:
    """Handles the merging of a PEFT adapter into a base transformer model."""
    
    def __init__(self, base_model_id: str, adapter_path: str, output_path: str):
        self.base_model_id = base_model_id
        self.adapter_path = adapter_path
        self.output_path = output_path
        self.model = None
        self.tokenizer = None

    def load_base_model(self):
        """Loads the base model and tokenizer."""
        logger.info(f"Loading base model from {self.base_model_id}...")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_id,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_id)
            logger.info("Base model and tokenizer loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load base model: {e}")
            raise

    def merge_adapter(self):
        """Merges the LoRA adapter into the base model."""
        if self.model is None:
            raise ValueError("Base model must be loaded before merging adapter.")
        
        logger.info(f"Loading LoRA adapter from {self.adapter_path}...")
        try:
            model = PeftModel.from_pretrained(self.model, self.adapter_path)
            logger.info("Merging adapter into base model...")
            self.model = model.merge_and_unload()
            logger.info("Merge completed successfully.")
        except Exception as e:
            logger.error(f"Failed to merge adapter: {e}")
            raise

    def save_merged_model(self):
        """Saves the merged model and tokenizer to the specified path."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be initialized before saving.")
        
        logger.info(f"Saving merged model to {self.output_path}...")
        try:
            self.model.save_pretrained(self.output_path)
            self.tokenizer.save_pretrained(self.output_path)
            logger.info("Custom model saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise

    def run(self):
        """Executes the full merge pipeline."""
        self.load_base_model()
        self.merge_adapter()
        self.save_merged_model()

if __name__ == "__main__":
    # Configuration
    CONFIG = {
        "base_model_id": "mistralai/Mistral-7B-Instruct-v0.2",
        "adapter_path": "./mistral-7b-tamsui-adapter/final_adapter",
        "merged_model_path": "./merged_mistral_tamsui_guide"
    }

    merger = ModelMerger(
        base_model_id=CONFIG["base_model_id"],
        adapter_path=CONFIG["adapter_path"],
        output_path=CONFIG["merged_model_path"]
    )
    
    try:
        merger.run()
    except Exception as e:
        logger.critical(f"Model merging process failed: {e}")
