import os
import json
import traceback
from typing import List, Dict, Callable
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from nodes.ocr_node import ocr_node
from nodes.extract_node import extract_node
from nodes.display_transactions_node import display_transactions_node
from nodes.get_human_input_node import get_human_input_node
from nodes.process_human_input_node import process_human_input_node
from nodes.check_if_done_node import check_if_done_node
from nodes.store_csv_node import store_csv_node
from utils.state import State

# Set up the LangGraph
workflow = StateGraph(State)

workflow.add_node("ocr", ocr_node)
workflow.add_node("extract", extract_node)
workflow.add_node("display_transactions", display_transactions_node)
workflow.add_node("get_human_input", get_human_input_node)
workflow.add_node("process_human_input", process_human_input_node)
workflow.add_node("check_if_done", check_if_done_node)
workflow.add_node("store_csv", store_csv_node)

workflow.set_entry_point("ocr")
workflow.add_edge("ocr", "extract")
workflow.add_edge("extract", "display_transactions")
workflow.add_edge("display_transactions", "get_human_input")
workflow.add_edge("get_human_input", "process_human_input")
workflow.add_edge("process_human_input", "check_if_done")
workflow.add_edge("store_csv", END)

# Add conditional edge
workflow.add_conditional_edges(
    "check_if_done",
    lambda state: "done" if state['is_done'] else "continue",
    {
        "continue": "display_transactions",
        "done": "store_csv"
    }
)

# Add Memory
memory = MemorySaver()

# Create the app
app = workflow.compile(checkpointer=memory, interrupt_before=["get_human_input"])
    
# Function to process a file through the entire workflow
def process_stream(file_path: str) -> List[Dict[str, str]]:
    thread = {"configurable": {"thread_id": "1"}}

    initial_state = State(file_path=file_path, ocr_text="", transactions=[],
                          human_input="", is_done=False, error="")
    for event in app.stream(initial_state, thread, stream_mode="values"):
        current_state=event
    while True:
        # Check if the stream is done
        if not current_state.get('is_done', False):
            # If not done, get human input
            human_input = input("\nEnter your changes (or 'ok' to finish): ").strip()
            thread = app.update_state(thread, {"human_input": human_input}, as_node="get_human_input")
        else:
            break
        # Continue the stream
        for event in app.stream(None, thread, stream_mode="values"):
            current_state=event
    return current_state['transactions']

# Process the file from UI
def process_file_from_ui(file_path: str, thread: Dict, human_input: str = None) -> List[Dict[str, str]]:
    initial_state = None if human_input is not None else State(file_path=file_path, ocr_text="", transactions=[],
                          human_input="", is_done=False, error="")
    if human_input is not None:
        app.update_state(thread, {"human_input": human_input}, as_node="get_human_input")
    for event in app.stream(initial_state, thread, stream_mode="values"):
        current_state=event
    return current_state['transactions']

# Example usage
if __name__ == "__main__":
    file_path = "./input/sample.pdf"
    result = process_stream(file_path)
    if result:
        print("Final processed transactions:")
        for transaction in result:
            print(json.dumps(transaction, indent=2))
    else:
        print("No transactions were processed due to an error.")