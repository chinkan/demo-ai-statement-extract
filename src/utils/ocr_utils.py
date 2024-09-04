import os
from google.cloud import vision
from google.cloud.vision_v1 import types
import io
from pdf2image import convert_from_path
import sys
from typing import List

def get_google_vision_client():
    # Initialize Google Cloud Vision client
    return vision.ImageAnnotatorClient()

def convert_pdf_to_images(pdf_path: str) -> List[bytes]:
    try:
        images = convert_from_path(pdf_path, poppler_path=os.getenv("POPPLER_PATH"))
        return [image_to_byte_array(img) for img in images]
    except Exception as e:
        print(f"Error converting PDF to images: {str(e)}")
        sys.exit(1)

def image_to_byte_array(image):
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, format='PNG')
    return imgByteArr.getvalue()

# Use pdf2image to perform OCR if the quota is not enough for cloud vision
def perform_ocr_pdf2image(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if file_path.lower().endswith('.pdf'):
        images = convert_pdf_to_images(file_path)
    else:
        with open(file_path, 'rb') as image_file:
            images = [image_file.read()]
    
    full_text = ""
    for image in images:
        image = types.Image(content=image)
        response = get_google_vision_client().document_text_detection(image=image)
        full_text += response.full_text_annotation.text + "\n\n"
    
    return full_text.strip()

def perform_ocr_pdf(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if not file_path.lower().endswith('.pdf'):
        raise ValueError(f"File {file_path} is not a PDF format")
    
    with open(file_path, "rb") as pdf_file:
        content = pdf_file.read()
    
    # Set InputConfig
    input_config = vision.InputConfig(
        content=content,
        mime_type="application/pdf"
    )

    # Set Feature
    feature = vision.Feature(
        type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION
    )

    # Set Request
    request = vision.AnnotateFileRequest(
        input_config=input_config,
        features=[feature]
    )

    # Execute request
    response = get_google_vision_client().batch_annotate_files(requests=[request])

    # Process response
    full_text = ""
    for res in response.responses[0].responses:
        full_text += res.full_text_annotation.text + "\n\n"
    
    if response.responses[0].error.message:
        raise Exception(
            f'{response.responses[0].error.message}\n'
            'For more information, please refer to: '
            'https://cloud.google.com/apis/design/errors'
        )
    
    return full_text.strip()