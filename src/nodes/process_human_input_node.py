from utils.human_input_utils import interpret_and_update_gemini
from utils.state import State
from utils.decorators import log_node_entry_exit
from langsmith import traceable

@log_node_entry_exit
@traceable(run_type="llm", name="process_human_input_node")
def process_human_input_node(state: State) -> State:
    if state['human_input'].lower() != 'ok':
        updated_transactions = interpret_and_update_gemini(state['human_input'], state['transactions'])
        if updated_transactions:
            state['transactions'] = updated_transactions
    return state