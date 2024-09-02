from typing import TypedDict, List, Dict

class State(TypedDict):
    file_path: str
    ocr_text: str
    transactions: List[Dict[str, str]]
    final_transactions: List[Dict[str, str]]
    human_input: str
    is_done: bool
    error: str