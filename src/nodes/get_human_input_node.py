from utils.state import State
from utils.decorators import log_node_entry_exit
from langsmith import traceable

@log_node_entry_exit
@traceable(run_type="chain", name="get_human_input_node")
def get_human_input_node(state: State) -> State:
    return state