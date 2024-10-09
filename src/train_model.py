import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 讀取CSV文件
df = pd.read_csv('./input/training_balanced.csv')
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
    return tokenizer(examples["Content"], truncation=True, padding="max_length", max_length=512, add_special_tokens=True)

# 對數據集應用預處理
tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_val = val_dataset.map(preprocess_function, batched=True)

# 設置模型
model = AutoModelForSequenceClassification.from_pretrained("HuggingFaceTB/SmolLM-135M", num_labels=2)
model.config.pad_token_id = tokenizer.pad_token_id # 確保模型知道pad_token_id

# 定義訓練參數
training_args = TrainingArguments(
    output_dir="./output/train_results",
    num_train_epochs=10,  # 增加訓練輪數
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=1000,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    eval_strategy="epoch",  # 定期評估
    save_strategy="epoch",
    load_best_model_at_end=True,  # 加載最佳模型
    # metric_for_best_model="eval_loss",  # 修改為"eval_loss"
)

# 設置Trainer並開始訓練
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
)

# 訓練模型
train_result = trainer.train()

# Evaluate the model by running the validation dataset from input/validation.csv
val_dataset = Dataset.from_pandas(pd.read_csv('./input/validation.csv'))
val_dataset = val_dataset.map(preprocess_function, batched=True)

# evaluate the model
eval_result = trainer.evaluate(eval_dataset=val_dataset)
print(eval_result)

# 獲取訓練歷史
history = trainer.state.log_history

# Plot training results
plt.figure(figsize=(10,6))

# Plot training loss and evaluation loss
train_loss = [x['loss'] for x in history if 'loss' in x]
eval_loss = [x['eval_loss'] for x in history if 'eval_loss' in x]

plt.plot(train_loss, label='Training Loss')
if eval_loss:
    plt.plot(range(0, len(eval_loss)*len(train_loss)//len(eval_loss), len(train_loss)//len(eval_loss)), eval_loss, label='Evaluation Loss')

plt.title('Training and Evaluation Loss')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()