from utils.ocr_utils import perform_ocr_pdf
from utils.state import State
from utils.decorators import log_node_entry_exit
from langsmith import traceable

@log_node_entry_exit
@traceable(run_type="chain", name="ocr_node")
def ocr_node(state: State) -> State:
    state['ocr_text'] = perform_ocr_pdf(state['file_path'])
    return state