import gradio as gr
import os
import json
from main import process_file_from_ui
import pandas as pd
import tempfile
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import uuid

def output_transactions(transactions):
    output.value = json.dumps(transactions, indent=2)

def process_statement(statement, thread, human_input = None):
    result = process_file_from_ui(statement, thread, human_input)
    return pd.DataFrame(result)

def process(statement, cloud_vision_api_key, openrouter_api_key, openrouter_model, openrouter_api_url):
    # 對於 gr.File 組件，我們直接獲取文件路徑
    statement_path = statement.name if hasattr(statement, 'name') else statement
    cloud_vision_api_key_path = cloud_vision_api_key.name if hasattr(cloud_vision_api_key, 'name') else cloud_vision_api_key

    # 讀取 cloud_vision_api_key 文件內容
    with open(cloud_vision_api_key_path, 'r') as key_file:
        cloud_vision_api_key_content = key_file.read()

    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.json') as temp_file:
        temp_file.write(cloud_vision_api_key_content)
        temp_file_path = temp_file.name

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_file_path
    os.environ["OPENROUTER_API_KEY"] = openrouter_api_key
    os.environ["OPENROUTER_MODEL"] = openrouter_model
    os.environ["OPENROUTER_API_URL"] = openrouter_api_url

    # 生成新的 thread ID
    thread = {"configurable": {"thread_id": str(uuid.uuid4())}}

    try:
        result = process_statement(statement_path, thread)
    finally:
        os.unlink(temp_file_path)

    return result, thread["configurable"]["thread_id"]

def continue_processing(human_input, thread_id):
    thread = {"configurable": {"thread_id": thread_id}}
    return process_statement(None, thread, human_input)

def export_transactions(output):
    output.to_csv("output.csv", index=False)
    return gr.File(value="output.csv", visible=True)

# 創建 FastAPI 應用
app = FastAPI()

# 添加 CORS 中間件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/process")
async def api_process(
    statement: UploadFile = File(...),
    cloud_vision_api_key: UploadFile = File(...),
    openrouter_api_key: str = Form(...),
    openrouter_model: str = Form(...),
    openrouter_api_url: str = Form(...)
):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_statement:
        temp_statement.write(await statement.read())
        temp_statement_path = temp_statement.name

    with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as temp_key:
        temp_key.write(await cloud_vision_api_key.read())
        temp_key_path = temp_key.name

    try:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_key_path
        os.environ["OPENROUTER_API_KEY"] = openrouter_api_key
        os.environ["OPENROUTER_MODEL"] = openrouter_model
        os.environ["OPENROUTER_API_URL"] = openrouter_api_url

        thread_id = str(uuid.uuid4())
        thread = {"configurable": {"thread_id": thread_id}}
        result = process_statement(temp_statement_path, thread)
        return {"result": result.to_dict(orient='records'), "thread_id": thread_id}
    finally:
        os.unlink(temp_statement_path)
        os.unlink(temp_key_path)

@app.post("/continue_processing")
async def api_continue_processing(human_input: str = Form(...), thread_id: str = Form(...)):
    thread = {"configurable": {"thread_id": thread_id}}
    result = continue_processing(human_input, thread_id)
    return result.to_dict(orient='records')

@app.post("/export_transactions")
async def api_export_transactions(output: str = Form(...)):
    df = pd.read_json(output)
    result = export_transactions(df)
    return {"file_path": result.value}

# Gradio 界面
with gr.Blocks() as demo:
    gr.Markdown("""
    # WizLedger - Monthly Statement Processor
    Processes statements using Google Cloud Vision API and OpenRouter API.
    Extracts transactions from monthly bank statements and exports them to CSV.
    """)
    
    with gr.Tabs():
        with gr.TabItem("Process Statement"):
            with gr.Row():
                with gr.Column(scale=1):   
                    statement = gr.File(label="Select a statement file")
                    process_button = gr.Button("Process")
                    output = gr.Dataframe(label="Output", interactive=False, value=[], headers=["date", "description", "amount"])
                    human_input = gr.Textbox(label="Human Input")
                    thread_id = gr.Textbox(label="Thread ID", interactive=False)
                    continue_button = gr.Button("Continue")
                with gr.Column(scale=1):
                    export_button = gr.Button("Export")
                    csv = gr.File(label="CSV", interactive=False)
            with gr.Row():
                with gr.Accordion("API Settings", open=True):
                    gr.Markdown("Google Cloud Vision API")
                    cloud_vision_api_key = gr.File(label="Select your Google Cloud Vision API key file")
                    gr.Markdown("OpenRouter API")
                    openrouter_api_key = gr.Textbox(label="Enter your OpenRouter API key", type="password", value=os.getenv("OPENROUTER_API_KEY"))
                    openrouter_model = gr.Textbox(label="Enter your OpenRouter model",value=os.getenv("OPENROUTER_MODEL"))
                    openrouter_api_url = gr.Textbox(label="Enter your OpenRouter API URL",value=os.getenv("OPENROUTER_API_URL"))
        
        with gr.TabItem("API Usage"):
            gr.Markdown("""
            # API Usage Examples
            
            Here are examples of how to use the API endpoints using Python's `requests` library:
            
            ```python
            import requests
            
            # Process a new statement
            url = "http://localhost:7860/process"
            files = {
                'statement': ('statement.pdf', open('path/to/statement.pdf', 'rb'), 'application/pdf'),
                'cloud_vision_api_key': ('key.json', open('path/to/cloud_vision_key.json', 'rb'), 'application/json')
            }
            data = {
                'openrouter_api_key': 'your_openrouter_api_key', # Your OpenRouter API key
                'openrouter_model': 'anthropic/claude-3.5-sonnet', # Recommend to use Claude 3.5 Sonnet, but you can use other models
                'openrouter_api_url': 'https://openrouter.ai/api/v1/chat/completions' # Compatible with OpenAI API
            }
            response = requests.post(url, files=files, data=data)
            result = response.json()
            print(result['result'])
            thread_id = result['thread_id']
            
            # Continue processing
            url = "http://localhost:7860/continue_processing"
            data = {
                'human_input': 'Some human input',
                'thread_id': thread_id
            }
            response = requests.post(url, data=data)
            print(response.json())
            response_data = response.json()
            
            # Export transactions
            url = "http://localhost:7860/export_transactions"
            data = {'output': response_data.data}
            response = requests.post(url, data=data)
            print(response.json())
            ```
            
            Make sure to replace the placeholder values with your actual data and API keys.
            """)

    def update_output_and_thread_id(result, thread_id):
        return result, thread_id

    process_button.click(process, 
                         inputs=[statement, cloud_vision_api_key, openrouter_api_key, openrouter_model, openrouter_api_url], 
                         outputs=[output, thread_id])
    continue_button.click(continue_processing, 
                          inputs=[human_input, thread_id], 
                          outputs=[output])
    export_button.click(export_transactions, inputs=[output], outputs=[csv])

# 將 Gradio 界面掛載到 FastAPI
app = gr.mount_gradio_app(app, demo, path="/")

# 啟動 FastAPI 服務器
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)