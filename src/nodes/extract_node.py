from utils.transaction_utils import extract_transactions_gemini
from utils.state import State
from utils.decorators import log_node_entry_exit
from langsmith import traceable

@log_node_entry_exit
@traceable(run_type="llm", name="extract_node")
def extract_node(state: State) -> State:
    # state['transactions'] = extract_transactions(state['ocr_text'])
    state['transactions'] = extract_transactions_gemini(state['ocr_text'])
    return state