import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
from sklearn.model_selection import train_test_split

# 讀取CSV文件
df = pd.read_csv('./input/data.csv')
df = df.rename(columns={'HasTransaction': 'labels'})

# 將數據分為訓練集和驗證集
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# 將DataFrame轉換為Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# 設置tokenizer
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
tokenizer.pad_token = tokenizer.eos_token # 將pad_token設置為eos_token

def preprocess_function(examples):
    return tokenizer(examples["Content"], truncation=True, padding="max_length", max_length=512)

# 對數據集應用預處理
tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_val = val_dataset.map(preprocess_function, batched=True)

# 設置模型
model = AutoModelForSequenceClassification.from_pretrained("HuggingFaceTB/SmolLM-135M", num_labels=2)
model.config.pad_token_id = tokenizer.pad_token_id # 確保模型知道pad_token_id

# 定義訓練參數
training_args = TrainingArguments(
    output_dir="./output/train_results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# 設置Trainer並開始訓練
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
)

trainer.train()

# 保存模型
trainer.save_model("./models/fine_tuned_model")