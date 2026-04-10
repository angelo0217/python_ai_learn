import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig, 
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_fine_tuning_config():
    """Centralized configuration for fine-tuning."""
    return {
        "base_model_id": "mistralai/Mistral-7B-Instruct-v0.2",
        "output_dir": "./mistral-7b-tamsui-adapter",
        "dataset_path": "./data.jsonl",
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    }

def get_quantization_config():
    """Returns the 4-bit quantization configuration for VRAM efficiency."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

def load_model_and_tokenizer(model_id):
    """Loads the model and tokenizer with quantization."""
    logger.info(f"Loading model from {model_id}...")
    bnb_config = get_quantization_config()
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.config.use_cache = False
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    return model, tokenizer

def prepare_lora_model(model, config):
    """Prepares the model for PEFT/LoRA training."""
    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    return model

def train():
    """Main training pipeline."""
    config = setup_fine_tuning_config()
    
    # 1. Load Model & Tokenizer
    model, tokenizer = load_model_and_tokenizer(config["base_model_id"])
    
    # 2. Prepare LoRA
    model = prepare_lora_model(model, config)
    
    # 3. Load Dataset
    logger.info(f"Loading dataset from {config['dataset_path']}...")
    dataset = load_dataset("json", data_files=config["dataset_path"], split="train")
    
    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        max_steps=100,
        save_strategy="steps",
        save_steps=50,
        fp16=False,
        bf16=True,
        optim="paged_adamw_32bit",
        report_to="none"
    )
    
    # 5. SFT Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text", # Ensure your jsonl has a 'text' field
        max_seq_length=2048,
        tokenizer=tokenizer,
        args=training_args,
    )
    
    logger.info("Starting training...")
    trainer.train()
    
    # 6. Save Adapter
    trainer.model.save_pretrained(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])
    logger.info(f"Model adapter saved to {config['output_dir']}")

if __name__ == "__main__":
    train()
