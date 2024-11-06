import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer
from datasets import Dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
import torch
import os

# 設置環境變量以啟用 MPS 後備
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# 檢查 MPS 是否可用
device = (
    "mps" 
    if torch.backends.mps.is_available() 
    else "cpu"
)
print(f"Using device: {device}")

# 讀取CSV文件
df = pd.read_csv('./input/training_balanced.csv')
df = df.rename(columns={'HasTransaction': 'labels'})

# 將數據分為訓練集和驗證集
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# 將DataFrame轉換為Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# 設置tokenizer
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_function(examples):
    # 為T5添加特定的前綴
    inputs = [f"classify transaction: {text}" for text in examples["Content"]]
    # 將標籤轉換為文本格式
    targets = [str(label) for label in examples["labels"]]
    
    model_inputs = tokenizer(
        inputs, 
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    # 對標籤進行編碼
    labels = tokenizer(
        targets,
        max_length=8,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# 對數據集應用預處理
tokenized_train = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
tokenized_val = val_dataset.map(preprocess_function, batched=True, remove_columns=val_dataset.column_names)

# 設置基礎模型
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
).to(device)

# 配置LoRA
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,  # T5是序列到序列模型
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q", "v"]  # T5的目標模塊
)

# 創建PEFT模型
model = get_peft_model(base_model, peft_config)
model.print_trainable_parameters()

# 定義訓練參數
training_args = TrainingArguments(
    output_dir="./output/lora_results",
    num_train_epochs=10,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    warmup_steps=500,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    auto_find_batch_size=True
)

# 設置Trainer並開始訓練
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
)

# 訓練前禁用快取以避免警告
model.config.use_cache = False

# 訓練模型
train_result = trainer.train()

# 評估模型
df_val = pd.read_csv('../input/validation.csv')
val_dataset = Dataset.from_pandas(df_val)
val_dataset = val_dataset.map(
    preprocess_function, 
    batched=True, 
    remove_columns=val_dataset.column_names
)
eval_result = trainer.evaluate(eval_dataset=val_dataset)
print(eval_result)

# 保存LoRA權重
model.save_pretrained("../output/lora_weights")
tokenizer.save_pretrained("../output/lora_weights")

# 繪製訓練結果
history = trainer.state.log_history

plt.figure(figsize=(10,6))

train_loss = [x['loss'] for x in history if 'loss' in x]
eval_loss = [x['eval_loss'] for x in history if 'eval_loss' in x]

plt.plot(train_loss, label='Training Loss')
if eval_loss:
    plt.plot(range(0, len(eval_loss)*len(train_loss)//len(eval_loss), len(train_loss)//len(eval_loss)), 
             eval_loss, label='Evaluation Loss')

plt.title('Training and Evaluation Loss')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
