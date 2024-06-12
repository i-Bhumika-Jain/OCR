from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import cv2
from sklearn.cluster import DBSCAN
import numpy as np
from pdf2image import convert_from_path
import os

app = FastAPI()

# Initialize PaddleOCR
paddleocr = PaddleOCR(lang="en", ocr_version="PP-OCRv4", show_log=False, use_gpu=True)

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

def process_receipt(receipt_texts, receipt_boxes, receipt_scores, receipt_results, output_json):
    item_mapping = {}
    for result in receipt_results:
        item_mapping[str(result[0])] = (str(result[1][0]), str(result[1][1]))

    average_y_coords = []
    for box_corners in receipt_boxes:
        y_coords = [corner[1] for corner in box_corners]
        average_y = sum(y_coords) / len(y_coords)
        average_y_coords.append(average_y)

    average_y_coords = np.array(average_y_coords).reshape(-1, 1)
    epsilon = 5
    min_samples = 1
    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples).fit(average_y_coords)
    labels = dbscan.labels_

    grouped_bounding_boxes = {}
    for i, label in enumerate(labels):
        if label not in grouped_bounding_boxes:
            grouped_bounding_boxes[label] = []
        grouped_bounding_boxes[label].append(receipt_boxes[i])

    for group_id, boxes in grouped_bounding_boxes.items():
        key_list = []
        for box in boxes:
            key_list.append(item_mapping[str(box)][0])
        main_key, main_value = '', []
        for key in key_list:
            if main_key == '' and isinstance(key, str):
                main_key = key
            else:
                main_value.append(key)
        if main_key == '':
            main_key = main_value[0]
            main_value = main_value[1:]
        if main_key in output_json:
            output_json[main_key].extend(main_value)
        else:
            output_json[main_key] = main_value

    total_amount_key = "TOTAL"
    if total_amount_key in output_json:
        total_amount_value = " ".join(output_json.pop(total_amount_key))
        output_json[total_amount_key] = total_amount_value

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
            output_json = process_receipt(receipt_texts, receipt_boxes, receipt_scores, receipt_results, output_json)
    else:
        receipt_texts, receipt_boxes, receipt_scores, receipt_results = paddle_scan(paddleocr, file_path)
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        font_path = './latin.ttf'
        annotated = draw_ocr(img, receipt_boxes, receipt_texts, receipt_scores, font_path=font_path)
        annotated_img = Image.fromarray(annotated)
        annotated_img.save('./ocr_visualisation.png')
        output_json = process_receipt(receipt_texts, receipt_boxes, receipt_scores, receipt_results, output_json)

    return output_json

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="10.41.26.40", port=8082)
