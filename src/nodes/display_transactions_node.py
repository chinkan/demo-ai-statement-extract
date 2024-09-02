from utils.state import State
from utils.decorators import log_node_entry_exit
import json

@log_node_entry_exit
def display_transactions_node(state: State) -> State:
    print("\nCurrent transactions:")
    for idx, transaction in enumerate(state['transactions']):
        print(f"{idx + 1}. {json.dumps(transaction)}")
    return state