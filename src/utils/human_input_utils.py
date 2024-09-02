import requests
import json
import os
from typing import List, Dict

def extract_json_from_string(s: str) -> str:
    """Extract JSON object from a string."""
    json_start = s.find('{')
    json_end = s.rfind('}') + 1
    if json_start != -1 and json_end != -1:
        return s[json_start:json_end]
    return ""

def interpret_and_update(user_input: str, transactions: List[Dict[str, str]]) -> List[Dict[str, str]]:
    prompt = f"""Given the following list of transactions:

{json.dumps(transactions, indent=2)}

And the user's input:

"{user_input}"

Please interpret the user's intention and provide the updated list of transactions. 
If the user wants to modify a specific transaction, update only that transaction.
If the user wants to add a new transaction, add it to the list.
If the user wants to delete a transaction, remove it from the list.

Provide ONLY the updated list of transactions in the following JSON format without any additional text:
{{
    "transactions": [
        {{"date": "DD MMM YYYY", "description": "Transaction description", "amount": "Amount in decimal format"}}
    ]
}}"""

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
        updated_transactions_str = result['choices'][0]['message']['content']
        
        print("AI Response:")
        print(updated_transactions_str)
        
        # Extract JSON from the response
        json_str = extract_json_from_string(updated_transactions_str)
        
        if not json_str:
            print("Error: No valid JSON found in the AI response.")
            return transactions
        
        try:
            updated_data = json.loads(json_str)
            if 'transactions' in updated_data and isinstance(updated_data['transactions'], list):
                return updated_data['transactions']
            else:
                print("Error: AI response does not contain a 'transactions' list.")
                return transactions
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON from AI response: {str(e)}")
            print("Extracted JSON content:")
            print(json_str)
            return transactions
    except requests.RequestException as e:
        print(f"Error making request to OpenRouter API: {str(e)}")
        return transactions
    except KeyError as e:
        print(f"Error accessing API response data: {str(e)}")
        print("API response content:")
        print(json.dumps(result, indent=2))
        return transactions
    except Exception as e:
        print(f"Unexpected error in interpret_and_update: {str(e)}")
        return transactions