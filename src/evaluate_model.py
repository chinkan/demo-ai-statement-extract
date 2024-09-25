import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 載入微調後的模型和tokenizer
model_path = "./models/fine_tuned_model"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 設置設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 讀取測試數據
def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

test_data = read_jsonl('./output/train_data/valid.jsonl')

# 準備預測
predictions = []
true_labels = []

# 進行預測
model.eval()
with torch.no_grad():
    for item in test_data:
        inputs = tokenizer(item['prompt'], return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        
        predictions.append(predicted_class)
        true_labels.append(int(item['completion']))

# 計算評估指標
accuracy = accuracy_score(true_labels, predictions)
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='binary')

# 輸出結果
print(f"準確率: {accuracy:.4f}")
print(f"精確率: {precision:.4f}")
print(f"召回率: {recall:.4f}")
print(f"F1分數: {f1:.4f}")