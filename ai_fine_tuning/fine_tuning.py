import torch
import gc
import logging
from typing import Dict, Any
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from core.config import settings
from core.exceptions import ModelError
from core.logger import logger

class FineTuningConfig:
    """
    Configuration for Fine-Tuning process.
    """
    def __init__(self):
        self.base_model_id = "mistralai/Mistral-7B-Instruct-v0.2"
        self.output_dir = "./mistral-7b-tamsui-adapter"
        self.dataset_path = "./data.jsonl"
        self.lora_r = 16
        self.lora_alpha = 32
        self.lora_dropout = 0.05
        self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        self.batch_size = 4
        self.gradient_accumulation_steps = 4
        self.learning_rate = 2e-4
        self.epochs = 3

def clear_gpu_memory():
    """
    Explicitly clear GPU cache and perform garbage collection.
    """
    logger.info("Clearing GPU memory...")
    torch.cuda.empty_cache()
    gc.collect()

def initialize_model_and_tokenizer(config: FineTuningConfig):
    """
    Initialize the quantized model and tokenizer.
    """
    try:
        logger.info(f"Loading model from {config.base_model_id}...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            config.base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
        )
        model.config.use_cache = False

        tokenizer = AutoTokenizer.from_pretrained(config.base_model_id)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        raise ModelError(f"Model initialization failed: {e}")

def setup_lora(model, config: FineTuningConfig):
    """
    Apply LoRA configuration to the model.
    """
    logger.info("Setting up LoRA configuration...")
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    return model

def load_and_validate_dataset(path: str):
    """
    Load dataset and perform basic validation.
    """
    try:
        logger.info(f"Loading dataset from {path}...")
        dataset = load_dataset("json", data_files=path, split="train")
        if len(dataset) == 0:
            raise ModelError("Dataset is empty.")
        logger.info(f"Dataset loaded successfully with {len(dataset)} samples.")
        return dataset
    except Exception as e:
        logger.error(f"Dataset loading failed: {str(e)}")
        raise ModelError(f"Dataset error: {e}")

def run_training(model, tokenizer, dataset, config: FineTuningConfig):
    """
    Execute the SFT training process.
    """
    try:
        training_args = TrainingArguments(
            output_dir=config.output_dir,
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            num_train_epochs=config.epochs,
            logging_steps=10,
            optim="paged_adamw_32bit",
            save_strategy="epoch",
            fp16=True,
            report_to="none"
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=512,
            args=training_args,
            tokenizer=tokenizer,
        )

        logger.info("Starting training...")
        trainer.train()
        trainer.save_model(config.output_dir)
        logger.info(f"Training completed. Model saved to {config.output_dir}")
    except torch.cuda.OutOfMemoryError:
        logger.error("CUDA Out of Memory error occurred during training.")
        clear_gpu_memory()
        raise ModelError("GPU OOM: Please reduce batch size or sequence length.")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise ModelError(f"Training error: {e}")

def main():
    config = FineTuningConfig()
    
    try:
        clear_gpu_memory()
        model, tokenizer = initialize_model_and_tokenizer(config)
        model = setup_lora(model, config)
        dataset = load_and_validate_dataset(config.dataset_path)
        
        run_training(model, tokenizer, dataset, config)
        
    except ModelError as me:
        logger.critical(f"Fine-tuning process aborted: {me}")
    except Exception as e:
        logger.critical(f"Unexpected error during fine-tuning: {e}")
    finally:
        clear_gpu_memory()

if __name__ == "__main__":
    main()
