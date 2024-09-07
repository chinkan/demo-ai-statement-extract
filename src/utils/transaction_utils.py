import requests
import json
import csv
import os
from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def extract_transactions(ocr_text: str) -> List[Dict[str, str]]:
    prompt = f"""You are an AI assistant trained to extract transaction information from financial statements. 
    Given the following text from a financial statement, please extract all transactions and format them as a list of JSON objects.
    Each transaction should have the following properties:
    - date: The date of the transaction in the format YYYY-MM-DD
    - description: A brief description of the transaction
    - amount: The transaction amount as a float (negative for debits, positive for credits)

    Here's the text from the financial statement:

    {ocr_text}

    Please return ONLY the list of JSON objects, without any additional explanation or text."""

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

def extract_transactions_locally(ocr_text: str) -> List[Dict[str, str]]:
    # 檢查系統係咪有 CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用緊嘅設備係: {device}")

    # Load the model and tokenizer
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    # Prepare the prompt
    prompt = f"""You are an AI assistant trained to extract transaction information from financial statements. 
    Given the following text from a financial statement, please extract all transactions and format them as a list of JSON objects.
    Each transaction should have the following properties:
    - date: The date of the transaction in the format YYYY-MM-DD
    - description: A brief description of the transaction
    - amount: The transaction amount as a float (negative for debits, positive for credits)

    Here's the text from the financial statement:

    {ocr_text}

    Please return ONLY the list of JSON objects, without any additional explanation or text."""

    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)

    # Generate the response
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=1024, num_return_sequences=1, temperature=0.7)

    # Decode the output
    transactions_str = tokenizer.decode(output[0], skip_special_tokens=True)

    try:
        # Parse the JSON string
        transactions = json.loads(transactions_str)

        # Validate and clean up the transactions
        validated_transactions = []
        for transaction in transactions:
            if all(key in transaction for key in ['date', 'description', 'amount']):
                # Ensure amount is a float
                transaction['amount'] = float(transaction['amount'])
                validated_transactions.append(transaction)

        return validated_transactions
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {str(e)}")
        return []
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

if __name__ == "__main__":  
    with open("output/sample.txt", "r", encoding="utf-8") as f:
        ocr_text = f.read()
    transactions = extract_transactions_locally(ocr_text)
    print(transactions)

    with open("output/transactions.csv", "w", encoding="utf-8") as f:
        f.write("date,description,amount\n")
        for transaction in transactions:
            f.write(f"{transaction['date']},{transaction['description']},{transaction['amount']}\n")