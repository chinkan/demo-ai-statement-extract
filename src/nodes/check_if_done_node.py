from utils.state import State
from utils.decorators import log_node_entry_exit
from langsmith import traceable

@log_node_entry_exit
@traceable(run_type="chain", name="check_if_done_node")
def check_if_done_node(state: State) -> State:
    state['is_done'] = state['human_input'].lower() == 'ok'
    return state