from utils.state import State
from utils.decorators import log_node_entry_exit
import json
from langsmith import traceable

@log_node_entry_exit
@traceable(run_type="chain", name="display_transactions_node")
def display_transactions_node(state: State) -> State:
    print("\nCurrent transactions:")
    for idx, transaction in enumerate(state['transactions']):
        print(f"{idx + 1}. {json.dumps(transaction)}")
    return state