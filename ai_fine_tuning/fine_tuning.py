import torch
import logging
import os
from typing import Optional
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FineTuningManager:
    """Handles the fine-tuning process for LLMs using PEFT/LoRA."""
    
    def __init__(self, base_model_id: str, output_dir: str, dataset_path: str):
        self.base_model_id = base_model_id
        self.output_dir = output_dir
        self.dataset_path = dataset_path
        self.model = None
        self.tokenizer = None

    def _get_bnb_config(self) -> BitsAndBytesConfig:
        """Returns 4-bit quantization configuration to save VRAM."""
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    def _get_lora_config(self) -> LoraConfig:
        """Returns LoRA configuration for Mistral models."""
        return LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )

    def setup_model_and_tokenizer(self):
        """Loads the model and tokenizer with quantization."""
        try:
            logger.info(f"Loading model from {self.base_model_id}...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_id,
                quantization_config=self._get_bnb_config(),
                device_map="auto",
            )
            self.model.config.use_cache = False

            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_id)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"
            
            logger.info("Model and tokenizer loaded successfully.")
        except torch.cuda.OutOfMemoryError:
            logger.error("GPU VRAM is insufficient. Please try a smaller model or reduce batch size.")
            raise
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def prepare_training(self):
        """Prepares the model for k-bit training and applies LoRA."""
        if self.model is None:
            raise RuntimeError("Model must be loaded before preparing training.")
        
        try:
            self.model = prepare_model_for_kbit_training(self.model)
            self.model = get_peft_model(self.model, self._get_lora_config())
            logger.info("Model prepared for PEFT training.")
        except Exception as e:
            logger.error(f"Error during PEFT preparation: {str(e)}")
            raise

    def run_training(self):
        """Executes the SFT training process."""
        try:
            if not os.path.exists(self.dataset_path):
                raise FileNotFoundError(f"Dataset file not found at {self.dataset_path}")

            dataset = load_dataset("json", data_files=self.dataset_path, split="train")
            
            training_args = TrainingArguments(
                output_dir=self.output_dir,
                per_device_train_batch_size=4,
                gradient_accumulation_steps=4,
                learning_rate=2e-4,
                logging_steps=10,
                max_steps=100,
                save_steps=50,
                fp16=True,
                optim="paged_adamw_32bit",
            )

            trainer = SFTTrainer(
                model=self.model,
                train_dataset=dataset,
                peft_config=self._get_lora_config(),
                dataset_text_field="text",
                max_seq_length=512,
                tokenizer=self.tokenizer,
                args=training_args,
            )

            logger.info("Starting training...")
            trainer.train()
            trainer.save_model(self.output_dir)
            logger.info(f"Training completed. Model saved to {self.output_dir}")

        except torch.cuda.OutOfMemoryError:
            logger.error("GPU VRAM exhausted during training. Reduce batch size or sequence length.")
            raise
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise

def main():
    # Configuration
    config = {
        "base_model_id": "mistralai/Mistral-7B-Instruct-v0.2",
        "output_dir": "./mistral-7b-tamsui-adapter",
        "dataset_path": "./data.jsonl"
    }

    manager = FineTuningManager(**config)
    
    try:
        manager.setup_model_and_tokenizer()
        manager.prepare_training()
        manager.run_training()
    except Exception as e:
        logger.critical(f"Application crashed: {str(e)}")

if __name__ == "__main__":
    main()
