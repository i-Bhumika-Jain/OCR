# this is giving date and time seprately

# import re
# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import JSONResponse
# from paddleocr import PaddleOCR, draw_ocr
# from PIL import Image
# import cv2
# import numpy as np
# from pdf2image import convert_from_path
# import os

# app = FastAPI()

# # Initialize PaddleOCR
# paddleocr = PaddleOCR(lang="en", ocr_version="PP-OCRv4", show_log=False, use_gpu=False)

# def paddle_scan(paddleocr, img_path_or_nparray):
#     result = paddleocr.ocr(img_path_or_nparray, cls=True)
#     result = result[0]
#     boxes = [line[0] for line in result]
#     txts = [line[1][0] for line in result]
#     scores = [line[1][1] for line in result]
#     return txts, boxes, scores, result

# def pdf_to_images(pdf_path):
#     images = convert_from_path(pdf_path)
#     return images

# def process_receipt(receipt_texts, receipt_boxes, receipt_scores, receipt_results):
#     output_json = {
#         "total_amount": "",
#         "date_of_bill": "",
#         "time_of_bill": "",
#         "categories": "",
#         "subcategories": "",
#         "payment_type": ""
#     }
    
#     categories = {
#         "Accommodation": ["hotel", "lodging", "airbnb"],
#         "Meals and Entertainment": ["food", "drinks", "beverages", "snacks", "restaurant", "dinner", "lunch", "fine dining", "bar", "pub", "catering", "takeout", "buffet", "happy hour", "entertainment", "live music", "theater", "movie tickets","table"],
#         "Transportation": ["taxi", "car rental", "bus", "train", "airport transfer", "shuttle", "rental agreement", "public transport", "route", "trip"],
#         "Fuel/Mileage": ["gasoline", "mileage", "gas station", "fuel pump", "vehicle", "odometer reading", "fuel efficiency"],
#         "Office Supplies": ["stationery", "printer cartridge"],
#         "Software & Subscriptions": ["software", "subscription", "license", "renewal", "online service", "cloud storage", "app", "renewal date"],
#         "Training & Development": ["course", "workshop", "seminar", "certification", "training"],
#         "Event & Conference": ["registration", "conference"],
#         "Equipment & Hardware": ["laptop", "desktop", "phone", "projector"],
#         "Miscellaneous": ["miscellaneous", "other", "unspecified", "miscellaneous charges"]
#     }

#     for text, box in zip(receipt_texts, receipt_boxes):
#         for key, kw_list in categories.items():
#             if any(kw in text.lower() for kw in kw_list):
#                 output_json["categories"] = key
#                 output_json["subcategories"] = kw_list
                
#         if "TOTAL" in text.upper():
#             numeric_value = re.findall(r'[\d.]+', text)
#             if numeric_value:
#                 output_json["total_amount"] = numeric_value[0]
        
#         # Combine detection and separation of date and time
#         match = re.search(r'(\d{2}-\d{2}-\d{4})(\d{2}:\d{2})?', text)
#         if match:
#             output_json["date_of_bill"] = match.group(1)
#             if match.group(2):
#                 output_json["time_of_bill"] = match.group(2)
#         elif re.match(r'\d{2}:\d{2}', text):
#             output_json["time_of_bill"] = text

#     return output_json

# @app.post("/processfile")
# async def process_file(file: UploadFile = File(...)):
#     file_location = f"./{file.filename}"

#     with open(file_location, "wb") as f:
#         f.write(await file.read())

#     output_json = process_uploaded_file(file_location)
#     os.remove(file_location)  # Clean up the file after processing
#     return JSONResponse(content=output_json)

# def process_uploaded_file(file_path: str):
#     output_json = {}

#     if file_path.endswith(".pdf"):
#         pdf_images = pdf_to_images(file_path)
#         for pdf_image in pdf_images:
#             pdf_image_np = np.array(pdf_image)
#             receipt_texts, receipt_boxes, receipt_scores, receipt_results = paddle_scan(paddleocr, pdf_image_np)
#             output_json.update(process_receipt(receipt_texts, receipt_boxes, receipt_scores, receipt_results))
#     else:
#         receipt_texts, receipt_boxes, receipt_scores, receipt_results = paddle_scan(paddleocr, file_path)
#         img = cv2.imread(file_path)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         font_path = './latin.ttf'
#         annotated = draw_ocr(img, receipt_boxes, receipt_texts, receipt_scores, font_path=font_path)
#         annotated_img = Image.fromarray(annotated)
#         annotated_img.save('./ocr_visualisation.png')
#         output_json.update(process_receipt(receipt_texts, receipt_boxes, receipt_scores, receipt_results))

#     return output_json

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="10.41.26.40", port=8082)




# iso format date only .........

#import re
# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import JSONResponse
# from paddleocr import PaddleOCR, draw_ocr
# from PIL import Image
# import cv2
# import numpy as np
# from pdf2image import convert_from_path
# import os
# from datetime import datetime

# app = FastAPI()

# # Initialize PaddleOCR
# paddleocr = PaddleOCR(lang="en", ocr_version="PP-OCRv4", show_log=False, use_gpu=False)

# def paddle_scan(paddleocr, img_path_or_nparray):
#     result = paddleocr.ocr(img_path_or_nparray, cls=True)
#     result = result[0]
#     boxes = [line[0] for line in result]
#     txts = [line[1][0] for line in result]
#     scores = [line[1][1] for line in result]
#     return txts, boxes, scores, result

# def pdf_to_images(pdf_path):
#     images = convert_from_path(pdf_path)
#     return images

# def process_receipt(receipt_texts, receipt_boxes, receipt_scores, receipt_results):
#     output_json = {
#         "total_amount": "",
#         "date_of_bill": "",
#         "time_of_bill": "",
#         "categories": "",
#         "subcategories": "",
#         "payment_type": ""
#     }
    
#     categories = {
#         "Accommodation": ["hotel", "lodging", "airbnb"],
#         "Meals and Entertainment": ["food", "drinks", "beverages", "snacks", "restaurant", "dinner", "lunch", "fine dining", "bar", "pub", "catering", "takeout", "buffet", "happy hour", "entertainment", "live music", "theater", "movie tickets","table"],
#         "Transportation": ["taxi", "car rental", "bus", "train", "airport transfer", "shuttle", "rental agreement", "public transport", "route", "trip"],
#         "Fuel/Mileage": ["gasoline", "mileage", "gas station", "fuel pump", "vehicle", "odometer reading", "fuel efficiency"],
#         "Office Supplies": ["stationery", "printer cartridge"],
#         "Software & Subscriptions": ["software", "subscription", "license", "renewal", "online service", "cloud storage", "app", "renewal date"],
#         "Training & Development": ["course", "workshop", "seminar", "certification", "training"],
#         "Event & Conference": ["registration", "conference"],
#         "Equipment & Hardware": ["laptop", "desktop", "phone", "projector"],
#         "Miscellaneous": ["miscellaneous", "other", "unspecified", "miscellaneous charges"]
#     }

#     for text, box in zip(receipt_texts, receipt_boxes):
#         for key, kw_list in categories.items():
#             if any(kw in text.lower() for kw in kw_list):
#                 output_json["categories"] = key
#                 output_json["subcategories"] = kw_list
                
#         if "TOTAL" in text.upper():
#             numeric_value = re.findall(r'[\d.]+', text)
#             if numeric_value:
#                 output_json["total_amount"] = numeric_value[0]
        
#         # Combine detection and separation of date and time
#         match = re.search(r'(\d{2}-\d{2}-\d{4})(\d{2}:\d{2})?', text)
#         if match:
#             date_str = match.group(1)
#             try:
#                 # Convert to datetime.date object
#                 date_obj = datetime.strptime(date_str, '%d-%m-%Y').date()
#                 # Serialize to ISO format (or any other desired format)
#                 output_json["date_of_bill"] = date_obj.isoformat()
#             except ValueError:
#                 output_json["date_of_bill"] = date_str  # Fallback to string if parsing fails
            
#             if match.group(2):
#                 output_json["time_of_bill"] = match.group(2)
#         elif re.match(r'\d{2}:\d{2}', text):
#             output_json["time_of_bill"] = text

#     return output_json

# @app.post("/processfile")
# async def process_file(file: UploadFile = File(...)):
#     file_location = f"./{file.filename}"

#     with open(file_location, "wb") as f:
#         f.write(await file.read())

#     output_json = process_uploaded_file(file_location)
#     os.remove(file_location)  # Clean up the file after processing
#     return JSONResponse(content=output_json)

# def process_uploaded_file(file_path: str):
#     output_json = {}

#     if file_path.endswith(".pdf"):
#         pdf_images = pdf_to_images(file_path)
#         for pdf_image in pdf_images:
#             pdf_image_np = np.array(pdf_image)
#             receipt_texts, receipt_boxes, receipt_scores, receipt_results = paddle_scan(paddleocr, pdf_image_np)
#             output_json.update(process_receipt(receipt_texts, receipt_boxes, receipt_scores, receipt_results))
#     else:
#         receipt_texts, receipt_boxes, receipt_scores, receipt_results = paddle_scan(paddleocr, file_path)
#         img = cv2.imread(file_path)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         font_path = './latin.ttf'
#         annotated = draw_ocr(img, receipt_boxes, receipt_texts, receipt_scores, font_path=font_path)
#         annotated_img = Image.fromarray(annotated)
#         annotated_img.save('./ocr_visualisation.png')
#         output_json.update(process_receipt(receipt_texts, receipt_boxes, receipt_scores, receipt_results))

#     return output_json

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="10.41.26.40", port=8082)


# date format is 2024-05-28T18:30:00.000Z

import re
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import cv2
import numpy as np
from pdf2image import convert_from_path
import os
from datetime import datetime, timedelta

app = FastAPI()

# Initialize PaddleOCR
paddleocr = PaddleOCR(lang="en", ocr_version="PP-OCRv4", show_log=False, use_gpu=False)

def paddle_scan(paddleocr, img_path_or_nparray):
    result = paddleocr.ocr(img_path_or_nparray, cls=True)
    result = result[0]
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    return txts, boxes, scores, result

def pdf_to_images(pdf_path):
    images = convert_from_path(pdf_path)
    return images

def process_receipt(receipt_texts, receipt_boxes, receipt_scores, receipt_results):
    output_json = {
        "total_amount": "",
        "date_of_bill": "",
        "time_of_bill": "",
        "categories": "",
        "subcategories": "",
        "payment_type": ""
    }
    
    categories = {
        "Accommodation": ["hotel", "lodging", "airbnb"],
        "Meals and Entertainment": ["food", "drinks", "beverages", "snacks", "restaurant", "dinner", "lunch", "fine dining", "bar", "pub", "catering", "takeout", "buffet", "happy hour", "entertainment", "live music", "theater", "movie tickets","table"],
        "Transportation": ["taxi", "car rental", "bus", "train", "airport transfer", "shuttle", "rental agreement", "public transport", "route", "trip"],
        "Fuel/Mileage": ["gasoline", "mileage", "gas station", "fuel pump", "vehicle", "odometer reading", "fuel efficiency"],
        "Office Supplies": ["stationery", "printer cartridge"],
        "Software & Subscriptions": ["software", "subscription", "license", "renewal", "online service", "cloud storage", "app", "renewal date"],
        "Training & Development": ["course", "workshop", "seminar", "certification", "training"],
        "Event & Conference": ["registration", "conference"],
        "Equipment & Hardware": ["laptop", "desktop", "phone", "projector"],
        "Miscellaneous": ["miscellaneous", "other", "unspecified", "miscellaneous charges"]
    }

    date_str = ""
    time_str = ""

    for text, box in zip(receipt_texts, receipt_boxes):
        for key, kw_list in categories.items():
            if any(kw in text.lower() for kw in kw_list):
                output_json["categories"] = key
                output_json["subcategories"] = kw_list
                
        if "TOTAL" in text.upper():
            numeric_value = re.findall(r'[\d.]+', text)
            if numeric_value:
                output_json["total_amount"] = numeric_value[0]
        
        # Combine detection and separation of date and time
        match = re.search(r'(\d{2}-\d{2}-\d{4})(\d{2}:\d{2})?', text)
        if match:
            date_str = match.group(1)
            if match.group(2):
                time_str = match.group(2)
        elif re.match(r'\d{2}:\d{2}', text):
            time_str = text

    if date_str:
        try:
            date_obj = datetime.strptime(date_str, '%d-%m-%Y')
            if time_str:
                time_obj = datetime.strptime(time_str, '%H:%M').time()
                date_obj = datetime.combine(date_obj, time_obj)
            else:
                date_obj = datetime.combine(date_obj, datetime.min.time())
            # Convert to ISO 8601 format with timezone information
            output_json["date_of_bill"] = date_obj.isoformat() + 'Z'
        except ValueError:
            output_json["date_of_bill"] = date_str  # Fallback to string if parsing fails
        
    if time_str and not output_json["date_of_bill"]:
        output_json["time_of_bill"] = time_str

    return output_json

@app.post("/processfile")
async def process_file(file: UploadFile = File(...)):
    file_location = f"./{file.filename}"

    with open(file_location, "wb") as f:
        f.write(await file.read())

    output_json = process_uploaded_file(file_location)
    os.remove(file_location)  # Clean up the file after processing
    return JSONResponse(content=output_json)

def process_uploaded_file(file_path: str):
    output_json = {}

    if file_path.endswith(".pdf"):
        pdf_images = pdf_to_images(file_path)
        for pdf_image in pdf_images:
            pdf_image_np = np.array(pdf_image)
            receipt_texts, receipt_boxes, receipt_scores, receipt_results = paddle_scan(paddleocr, pdf_image_np)
            output_json.update(process_receipt(receipt_texts, receipt_boxes, receipt_scores, receipt_results))
    else:
        receipt_texts, receipt_boxes, receipt_scores, receipt_results = paddle_scan(paddleocr, file_path)
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        font_path = './latin.ttf'
        annotated = draw_ocr(img, receipt_boxes, receipt_texts, receipt_scores, font_path=font_path)
        annotated_img = Image.fromarray(annotated)
        annotated_img.save('./ocr_visualisation.png')
        output_json.update(process_receipt(receipt_texts, receipt_boxes, receipt_scores, receipt_results))

    return output_json

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="10.41.26.40", port=8082)
