import requests
import json
import csv
import os
from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from accelerate import Accelerator

def get_prompt(ocr_text: str) -> str:
    return f"""You are an AI assistant trained to extract transaction information from financial statements. 
    Given the following text from a financial statement, please extract all transactions and format them as a list of JSON objects.
    Each transaction should have the following properties:
    - date: The date of the transaction in the format YYYY-MM-DD
    - description: A brief description of the transaction
    - amount: The transaction amount as a float (negative for debits, positive for credits)

    Here's the text from the financial statement:

    {ocr_text}

    Please return ONLY the simplified list of JSON objects, without any additional explanation or text."""

def extract_transactions(ocr_text: str) -> List[Dict[str, str]]:
    prompt = get_prompt(ocr_text)

    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json"
    }

    data = {
        "model": os.getenv('OPENROUTER_MODEL'),
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        response = requests.post(os.getenv('OPENROUTER_API_URL'), headers=headers, json=data)
        response.raise_for_status()
        
        result = response.json()
        transactions_str = result['choices'][0]['message']['content']
        
        transactions = json.loads(transactions_str)
        
        # Validate and clean up the transactions
        validated_transactions = []
        for transaction in transactions:
            if all(key in transaction for key in ['date', 'description', 'amount']):
                # Ensure amount is a float
                transaction['amount'] = float(transaction['amount'])
                validated_transactions.append(transaction)
        
        return validated_transactions
    except Exception as e:
        print(f"Error extracting transactions: {str(e)}")
        return []

def store_transactions_csv(transactions: List[Dict[str, str]], filename: str):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['date', 'description', 'amount']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for transaction in transactions:
            writer.writerow(transaction)
    
    print(f"Transactions stored in {filename}")

def split_text_into_chunks(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks

def detect_has_transactions(chunk: str) -> bool:
    global model, tokenizer
    try:
        model, tokenizer
    except:
        # model_name = "./models/fine_tuned_model"  
        model_name = "HuggingFaceTB/SmolLM-135M" # Example of a very small model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

    inputs = tokenizer(f"Does this text contain financial transactions? {chunk}", return_tensors="pt", max_length=1024, truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1).item()
        return predicted_class == 1  # Assuming 1 means "yes"
    
def extract_transactions_locally(ocr_text: str) -> List[Dict[str, str]]:
    accelerator = Accelerator()
    prompt = get_prompt(ocr_text)
    model_name = "codellama/CodeLlama-7b-Instruct-hf" # Use medium model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model = accelerator.prepare(model)
    inputs = tokenizer(prompt, return_tensors="pt").to(accelerator.device)

    token_count = len(inputs['input_ids'])
    print(f"Token count: {token_count}")

    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=2048)
        transactions_str = tokenizer.batch_decode(outputs[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
        print(transactions_str)
        transactions = json.loads(transactions_str)
        return transactions

            
if __name__ == "__main__":  
    chunk_results = []
    for i in range(1, 3):
        with open(f"output/test{i}.txt", "r", encoding="utf-8") as file:
            ocr_text = file.read()
            chunks = split_text_into_chunks(ocr_text)
            
            for chunk in chunks:
                result = detect_has_transactions(chunk)
                chunk_results.append(["Does this text contain financial transactions? ", chunk, result])
            
    with open(f"output/result.csv", "w", newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Instruction", "Chunk", "Contains Transactions"])  # 寫入標題行
        writer.writerows(chunk_results)  # 寫入所有結果行

    # for i in range(1, 7):   
    #     with open(f"output/sample{i}.txt", "r", encoding="utf-8") as file:
    #         ocr_text = file.read()
    #         transactions = extract_transactions_locally(ocr_text)
    #         print(transactions)

