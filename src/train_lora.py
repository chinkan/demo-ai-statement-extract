import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# 讀取數據
data = pd.read_csv('./input/data.csv')
dataset = Dataset.from_pandas(data)

# 加載 tokenizer 和模型
model_name = "HuggingFaceTB/SmolLM-1.7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 設置 pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 預處理數據
def preprocess_function(examples):
    return tokenizer(examples['Content'], truncation=True, padding=True)

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# 訓練參數
training_args = TrainingArguments(
    output_dir='./output/lora',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# 訓練
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
)

trainer.train()
