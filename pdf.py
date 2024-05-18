from paddleocr import PaddleOCR, draw_ocr
import json
import argparse
from PIL import Image
import cv2
from sklearn.cluster import DBSCAN
import numpy as np
from pdf2image import convert_from_path

paddleocr = PaddleOCR(lang="en", ocr_version="PP-OCRv4", show_log=False, use_gpu=True)

def paddle_scan(paddleocr, img_path_or_nparray):
    result = paddleocr.ocr(img_path_or_nparray, cls=True)
    result = result[0]
    boxes = [line[0] for line in result]       # Bounding box
    txts = [line[1][0] for line in result]     # Raw text
    scores = [line[1][1] for line in result]   # Scores
    return txts, boxes, scores, result

def pdf_to_images(pdf_path):
    images = convert_from_path(pdf_path)
    return images

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, default="./food.png")  # Change argument name to file_path
    parser.add_argument("--output", type=str, default="output.json")
    args = parser.parse_args()

    # If processing PDF:
    if args.file_path.endswith(".pdf"):
        # Convert PDF to images
        pdf_images = pdf_to_images(args.file_path)
        output_json = {}
        # Process each page separately
        for i, pdf_image in enumerate(pdf_images):
            # Convert PIL Image to numpy array
            pdf_image_np = np.array(pdf_image)
            # Perform OCR scan
            receipt_texts, receipt_boxes, receipt_scores, receipt_results = paddle_scan(paddleocr, pdf_image_np)
            # Rest of the code remains the same
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

    else:
        # If processing image:
        receipt_texts, receipt_boxes, receipt_scores, receipt_results = paddle_scan(paddleocr, args.file_path)
        # Rest of the code remains the same
        font_path = './latin.ttf'
        img = cv2.imread(args.file_path)

        # Reorders the color channels
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Draw annotations on image
        annotated = draw_ocr(img, receipt_boxes, receipt_texts, receipt_scores, font_path=font_path)

        # Show the image using matplotlib
        annotated_img = Image.fromarray(annotated)
        annotated_img.save('./ocr_visualisation.png')

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
                if main_key == '' and isinstance(key, str):
                    main_key = key
                else:
                    main_value.append(key)
            if main_key == '':
                main_key = main_value[0]
                main_value = main_value[1:]
            output_json[main_key] = main_value

    print(output_json)

    with open(args.output, "w") as outfile:
        json.dump(output_json, outfile)
