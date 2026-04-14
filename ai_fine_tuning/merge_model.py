import torch
import os
import logging
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelMergeManager:
    """Handles the merging of LoRA adapters into base LLM models."""
    
    def __init__(self, base_model_id: str, adapter_path: str, merged_model_path: str):
        self.base_model_id = base_model_id
        self.adapter_path = adapter_path
        self.merged_model_path = merged_model_path
        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None

    def validate_paths(self) -> bool:
        """Check if required paths exist before proceeding."""
        if not os.path.exists(self.adapter_path):
            logger.error(f"Adapter path not found: {self.adapter_path}")
            return False
        return True

    def load_and_merge(self):
        """Loads base model and adapter, then merges them."""
        try:
            if not self.validate_paths():
                raise FileNotFoundError(f"Required adapter path {self.adapter_path} is missing.")

            logger.info(f"Loading base model from {self.base_model_id}...")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_id,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_id)

            logger.info(f"Loading LoRA adapter from {self.adapter_path}...")
            model = PeftModel.from_pretrained(base_model, self.adapter_path)

            logger.info("Merging model weights...")
            self.model = model.merge_and_unload()
            logger.info("Merge completed successfully.")

        except torch.cuda.OutOfMemoryError:
            logger.error("GPU Out of Memory during model merge. Try reducing batch size or using a larger GPU.")
            raise
        except Exception as e:
            logger.exception(f"An unexpected error occurred during merge: {e}")
            raise

    def save_merged_model(self):
        """Saves the merged model and tokenizer to the specified path."""
        if self.model is None or self.tokenizer is None:
            logger.error("No merged model or tokenizer available to save.")
            return

        try:
            logger.info(f"Saving merged model to {self.merged_model_path}...")
            self.model.save_pretrained(self.merged_model_path)
            self.tokenizer.save_pretrained(self.merged_model_path)
            logger.info("Custom model saved successfully.")
        except Exception as e:
            logger.exception(f"Failed to save merged model: {e}")
            raise

def main():
    # Configuration
    config = {
        "base_model_id": "mistralai/Mistral-7B-Instruct-v0.2",
        "adapter_path": "./mistral-7b-tamsui-adapter/final_adapter",
        "merged_model_path": "./merged_mistral_tamsui_guide"
    }

    manager = ModelMergeManager(**config)
    try:
        manager.load_and_merge()
        manager.save_merged_model()
    except Exception as e:
        logger.error(f"Model merge process failed: {e}")

if __name__ == "__main__":
    main()
