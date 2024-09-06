from utils.transaction_utils import extract_transactions, extract_transactions_locally
from utils.state import State
from utils.decorators import log_node_entry_exit

@log_node_entry_exit
def extract_node(state: State) -> State:
    # state['transactions'] = extract_transactions(state['ocr_text'])
    state['transactions'] = extract_transactions_locally(state['ocr_text'])
    return state