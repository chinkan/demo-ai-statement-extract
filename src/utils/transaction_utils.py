import requests
import json
import csv
import os
from typing import List, Dict

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

def store_transactions_csv(transactions: List[Dict[str, str]], filename: str):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['date', 'description', 'amount']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for transaction in transactions:
            writer.writerow(transaction)
    
    print(f"Transactions stored in {filename}")