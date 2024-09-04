from utils.state import State
from utils.decorators import log_node_entry_exit

@log_node_entry_exit
def get_human_input_node(state: State) -> State:
    return state