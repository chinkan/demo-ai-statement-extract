from utils.transaction_utils import store_transactions_csv
from utils.state import State
from typing import Annotated
from langgraph.graph import END
from utils.decorators import log_node_entry_exit
from langsmith import traceable

@log_node_entry_exit
@traceable(run_type="chain", name="store_csv_node")
def store_csv_node(state: State) -> Annotated[State, END]:
    if not state.get('error'):
        csv_filename = 'output/transactions.csv'
        store_transactions_csv(state['transactions'], csv_filename)
    return state