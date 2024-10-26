import requests
import json
import csv
import os
from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, RootModel

class Transaction(BaseModel):
    date: str = Field(description="The date of the transaction in the format YYYY-MM-DD")
    description: str = Field(description="A brief description of the transaction")
    amount: float = Field(description="The transaction amount as a float (negative for debits, positive for credits)")

class Transactions(RootModel[List[Transaction]]):
    pass

def get_prompt_template() -> str:
    return """You are an AI assistant trained to extract transaction information from financial statements. 
    Given the following text from a financial statement, please extract all transactions and format them as a list of JSON objects.
    Each transaction should have the following properties:
    - date: The date of the transaction in the format YYYY-MM-DD
    - description: A brief description of the transaction
    - amount: The transaction amount as a float (negative for debits, positive for credits)
    
    Here's the text from the financial statement:

    {ocr_text}
    
    Please return ONLY the list of JSON objects, without any additional explanation or text."""

def extract_transactions(ocr_text: str) -> List[Dict[str, str]]:
    prompt_text = get_prompt_template()

    llm = ChatOpenAI(
        model_name=os.getenv('OPENROUTER_MODEL'), 
        openai_api_base=os.getenv('OPENROUTER_API_URL'), 
        openai_api_key=os.getenv('OPENROUTER_API_KEY'))
    prompt = ChatPromptTemplate.from_template(prompt_text)
    parser = PydanticOutputParser(pydantic_object=Transactions)

    #*** OpenRouter not support structured output ***  
    # structured_llm = llm.with_structured_output(Transaction)
    # chain = prompt | structured_llm

    chain = prompt | llm | parser

    try:
        transactions = chain.invoke({"ocr_text": ocr_text})

        # validate and clean up the transactions
        validated_transactions = []
        for transaction in transactions.root:
            # because Transaction is a BaseModel, so we can access the properties directly
            if transaction.date and transaction.description and transaction.amount is not None:
                # ensure amount is a float
                transaction.amount = float(transaction.amount)
                validated_transactions.append(transaction.model_dump())
        
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

if __name__ == "__main__":

    # load_dotenv()

    print(os.getenv('OPENROUTER_API_KEY'))
    print(os.getenv('OPENROUTER_API_URL'))
    print(os.getenv('OPENROUTER_MODEL'))

    # Read the sample text file
    with open("output/sample1.txt", "r", encoding="utf-8") as file:
        ocr_text = file.read()
        transactions = extract_transactions(ocr_text)
        print(transactions)
