import traceback
from utils.state import State

def log_node_entry_exit(func):
    def wrapper(state: State) -> State:
        print(f"Entering {func.__name__}")
        try:
            result = func(state)
            print(f"Exiting {func.__name__} successfully")
            return result
        except Exception as e:
            print(f"Error in {func.__name__}: {str(e)}")
            error_state = State(file_path=state['file_path'], ocr_text=state.get('ocr_text', ''), 
                                transactions=state.get('transactions', []), 
                                final_transactions=state.get('final_transactions', []),
                                error=f"Error in {func.__name__}: {str(e)}\n{traceback.format_exc()}")
            return error_state
    return wrapper