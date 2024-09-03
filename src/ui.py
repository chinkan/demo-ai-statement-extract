import gradio as gr
import os
from main import process_statement

def greet(statement, cloud_vision_api_key, openrouter_api_key, openrouter_model, openrouter_api_url):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cloud_vision_api_key
    os.environ["OPENROUTER_API_KEY"] = openrouter_api_key
    os.environ["OPENROUTER_MODEL"] = openrouter_model
    os.environ["OPENROUTER_API_URL"] = openrouter_api_url
    return process_statement(statement)

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
            output = gr.Textbox(label="Output")
    with gr.Row():
        with gr.Accordion("API Settings", open=True):
            gr.Markdown("Google Cloud Vision API")
            cloud_vision_api_key = gr.File(label="Select your Google Cloud Vision API key")
            gr.Markdown("OpenRouter API")
            openrouter_api_key = gr.Textbox(label="Enter your OpenRouter API key", type="password")
            openrouter_model = gr.Textbox(label="Enter your OpenRouter model",value="anthropic/claude-3.5-sonnet")
            openrouter_api_url = gr.Textbox(label="Enter your OpenRouter API URL",value="https://openrouter.ai/api/v1/chat/completions")

demo.launch()