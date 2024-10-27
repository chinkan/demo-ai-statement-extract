from pydantic import BaseModel, Field, RootModel
from typing import List

class Transaction(BaseModel):
    date: str = Field(description="The date of the transaction in the format YYYY-MM-DD")
    description: str = Field(description="A brief description of the transaction")
    amount: float = Field(description="The transaction amount as a float (negative for debits, positive for credits)")

class Transactions(RootModel[List[Transaction]]):
    pass