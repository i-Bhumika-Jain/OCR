from paddleocr import PaddleOCR, draw_ocr
import json
import argparse
from PIL import Image
 
import cv2
 
from sklearn.cluster import DBSCAN
import numpy as np
import gradio as gr
 
 
 
 
 
def paddle_scan(paddleocr,img_path_or_nparray):
    result = paddleocr.ocr(img_path_or_nparray,cls=True)
    result = result[0]
    boxes = [line[0] for line in result]       #boundign box
    txts = [line[1][0] for line in result]     #raw text
    scores = [line[1][1] for line in result]   # scores
    return  txts, boxes, scores, result
 
# perform ocr scan
def ocr_function (image_pathS):
    image_path = image_pathS[0]
    paddleocr = PaddleOCR(lang="en",ocr_version="PP-OCRv4",show_log = False,use_gpu=True)
 
    receipt_texts, receipt_boxes, receipt_scores, receipt_results = paddle_scan(paddleocr,image_path)
 
    font_path = './latin.ttf'
    img = cv2.imread(image_path)
 
    # reorders the color channels
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 
 
    # draw annotations on image
    annotated = draw_ocr(img, receipt_boxes, receipt_texts, receipt_scores, font_path=font_path)
 
    annotated_img = Image.fromarray(annotated)
    # annotated_img.save('./ocr_visualisation.png')
 
    item_mapping = {}
    for result in receipt_results:
        item_mapping[str(result[0])] = (str(result[1][0]), str(result[1][1]))
 
 
    # Calculate the average y-coordinate for each bounding box
    average_y_coords = []
    for box_corners in receipt_boxes:
        # Extract y-coordinates from each corner
        y_coords = [corner[1] for corner in box_corners]
        # Calculate the average y-coordinate
        average_y = sum(y_coords) / len(y_coords)
        average_y_coords.append(average_y)
 
    # Convert the list of average y-coordinates to a numpy array
    average_y_coords = np.array(average_y_coords).reshape(-1, 1)
 
    # Apply DBSCAN clustering
    epsilon = 5  # Distance threshold
    min_samples = 1  # Minimum number of samples in a cluster
    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples).fit(average_y_coords)
 
    # Get the labels assigned to each bounding box by DBSCAN
    labels = dbscan.labels_
 
    # Group bounding boxes based on their labels
    grouped_bounding_boxes = {}
    for i, label in enumerate(labels):
        if label not in grouped_bounding_boxes:
            grouped_bounding_boxes[label] = []
        grouped_bounding_boxes[label].append(receipt_boxes[i])
 
    # Print the grouped bounding boxes
    output_json = {}
    for group_id, boxes in grouped_bounding_boxes.items():
        key_list = []
        for box in boxes:
            key_list.append(item_mapping[str(box)][0])
        main_key, main_value = '', []
        for key in key_list:
            if main_key=='' and isinstance(key, str):
                main_key = key
            else:
                main_value.append(key)
        if main_key == '':
            main_key = main_value[0]
            main_value = main_value[1:]
        output_json[main_key] = main_value
    print(output_json)
 
 
    # with open(output, "w") as outfile:
    #     json.dump(output_json, outfile)
 
    return output_json, annotated_img
 
if __name__ == '__main__':
   
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--image_path", type=str, default="./food.png")
    # parser.add_argument("--output", type=str, default="output.json")
    # args = parser.parse_args()
 
    # annotated_img, output_json, receipt_results = ocr_function (paddleocr, args.image_path, args.output)
 
    with gr.Blocks(theme="shivi/calm_seafoam", title='OCR Tool') as demo:
 
        # gr.HTML('''<img src="https://www.brandcrowd.com/maker/social/d22y12stzd" alt="OcR Logo" width="250" height="50" border="0"></a>''')
        with gr.Tab("OCR"):
            with gr.Row():
                with gr.Column():
                    files = gr.File(label = "Upload your bill image here", file_count='multiple', height=720)
                    submit = gr.Button("Generate OCR Results")
                with gr.Column():
                    schema = gr.JSON(label = "Arrangement JSON")
            with gr.Row():
                detected_image = gr.Image()
            submit.click(ocr_function, inputs = [files], outputs = [schema, detected_image])
 
    demo.queue(max_size=5)
    demo.launch(share=True, debug=True)