import requests
import json
import os
from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from models.transactions import Transactions

def get_prompt_template() -> str:
    return """Given the following list of transactions:

{transactions}

And the user's input:

"{user_input}"

Please interpret the user's intention and provide the updated list of transactions. 
- If the user wants to modify a specific transaction, update only that transaction.
- If the user wants to add a new transaction, add it to the list.
- If the user wants to delete a transaction, remove it from the list.

Each transaction should have the following properties:
    - date: The date of the transaction in the format YYYY-MM-DD
    - description: A brief description of the transaction
    - amount: The transaction amount as a float (negative for debits, positive for credits)

Provide ONLY the updated list of transactions in the JSON format.
Please return ONLY the list of JSON objects, without any additional explanation or text."""

def interpret_and_update(user_input: str, transactions: List[Dict[str, str]]) -> List[Dict[str, str]]:
    
    llm = ChatOpenAI(
        model_name=os.getenv('OPENROUTER_MODEL'), 
        openai_api_base=os.getenv('OPENROUTER_API_URL'), 
        openai_api_key=os.getenv('OPENROUTER_API_KEY'))
    
    prompt_text = get_prompt_template()
    prompt = ChatPromptTemplate.from_template(prompt_text)
    parser = PydanticOutputParser(pydantic_object=Transactions)

    chain = prompt | llm | parser

    try:
        updated_transactions: Transactions = chain.invoke({"transactions": transactions, "user_input": user_input})
        return updated_transactions.model_dump()
    except Exception as e:
        print(f"Unexpected error in interpret_and_update: {str(e)}")
        return transactions
    
if __name__ == "__main__":
    with open("output/transactions.json", "r", encoding="utf-8") as jsonfile:
        transactions = json.load(jsonfile)
        updated_transactions = interpret_and_update("Add a new transaction on 2024-08-11 for 100.00, description is 'test'", transactions)
        print(updated_transactions)
