import os
import json
import traceback
from typing import List, Dict
from langgraph.graph import Graph, END
from nodes.ocr_node import ocr_node
from nodes.extract_node import extract_node
from nodes.display_transactions_node import display_transactions_node
from nodes.get_human_input_node import get_human_input_node
from nodes.process_human_input_node import process_human_input_node
from nodes.check_if_done_node import check_if_done_node
from nodes.store_csv_node import store_csv_node
from utils.state import State

# Set up the LangGraph
workflow = Graph()

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

# Add conditional edge
workflow.add_conditional_edges(
    "check_if_done",
    lambda state: "done" if state['is_done'] else "continue",
    {
        "continue": "display_transactions",
        "done": "store_csv"
    }
)

# Create the app
app = workflow.compile()

# Function to process a file through the entire workflow
def process_statement(file_path: str) -> List[Dict[str, str]]:
    initial_state = State(file_path=file_path, ocr_text="", transactions=[], final_transactions=[], 
                          human_input="", is_done=False, error="")
    try:
        result = app.invoke(initial_state)
        if result is None:
            print("Error: Workflow returned None. This might indicate an issue in one of the nodes.")
            return []
        if result.get('error'):
            print(f"Error occurred during workflow execution: {result['error']}")
            return []
        if 'transactions' not in result:
            print("Error: 'transactions' not found in the workflow result.")
            print("Workflow result:")
            print(json.dumps(result, indent=2))
            return []
        return result['transactions']
    except Exception as e:
        print(f"An error occurred while processing the statement: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        return []

# Example usage
if __name__ == "__main__":
    file_path = "./input/sample.pdf"
    result = process_statement(file_path)
    if result:
        print("Final processed transactions:")
        for transaction in result:
            print(json.dumps(transaction, indent=2))
    else:
        print("No transactions were processed due to an error.")