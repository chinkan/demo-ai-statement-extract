import gradio as gr
import os
import json
from main import process_file_from_ui

thread = {"configurable": {"thread_id": "1"}} # TODO: make this thread_id configurable

def output_transactions(transactions):
    output.value = json.dumps(transactions, indent=2)

def process(statement, cloud_vision_api_key, openrouter_api_key, openrouter_model, openrouter_api_url):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cloud_vision_api_key
    os.environ["OPENROUTER_API_KEY"] = openrouter_api_key
    os.environ["OPENROUTER_MODEL"] = openrouter_model
    os.environ["OPENROUTER_API_URL"] = openrouter_api_url
    return process_file_from_ui(statement, thread)

def continue_processing(human_input):
    return process_file_from_ui(None, thread, human_input)

with gr.Blocks() as demo:
    gr.Markdown("""
    # Monthly Statement Processor
    Processes statements using OCR and AI.
    """)
    with gr.Row():
        with gr.Column(scale=1):   
            statement = gr.File(label="Select a statement file")
            process_button = gr.Button("Process")
        with gr.Column(scale=1):
            output = gr.Textbox(label="Output", interactive=False)
            human_input = gr.Textbox(label="Human Input")
            continue_button = gr.Button("Continue")
    with gr.Row():
        with gr.Accordion("API Settings", open=True):
            gr.Markdown("Google Cloud Vision API")
            cloud_vision_api_key = gr.File(label="Select your Google Cloud Vision API key", value=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
            gr.Markdown("OpenRouter API")
            openrouter_api_key = gr.Textbox(label="Enter your OpenRouter API key", type="password", value=os.getenv("OPENROUTER_API_KEY"))
            openrouter_model = gr.Textbox(label="Enter your OpenRouter model",value=os.getenv("OPENROUTER_MODEL"))
            openrouter_api_url = gr.Textbox(label="Enter your OpenRouter API URL",value=os.getenv("OPENROUTER_API_URL"))

    process_button.click(process, inputs=[statement, cloud_vision_api_key, openrouter_api_key, openrouter_model, openrouter_api_url], outputs=[output])
    continue_button.click(continue_processing, inputs=[human_input], outputs=[output])

demo.launch()