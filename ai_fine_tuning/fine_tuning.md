uv pip install transformers datasets peft accelerate bitsandbytes trl

# 2. 運行微調
python fine_tuning.py

# 3. 合併模型
python merge_model.py

# 4. 創建 Ollama 模型
ollama create my-python-helper -f ./Modelfile

相關存放位置 C:\Users\USER\.cache\huggingface