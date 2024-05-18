### **OCR Tool with PaddleOCR**
This repository provides a Python-based OCR (Optical Character Recognition) tool utilizing the PaddleOCR library. The tool is capable of processing both images and PDFs to extract text and organizes the extracted text using DBSCAN clustering.

Key Features
OCR Processing: Utilizes PaddleOCR to perform text detection, recognition, and classification.
PDF Support: Converts PDF documents to images and processes each page individually.
Clustering: Groups text elements based on their spatial distribution using the DBSCAN clustering algorithm.
Annotation: Generates annotated images with detected text boxes and recognized text.
JSON Output: Outputs the recognized and clustered text data in JSON format.
Gradio Interface: Provides a user-friendly web interface for uploading and processing images through Gradio.
Requirements
To run this project, you'll need the following dependencies:

Python 3.6+
PaddleOCR
OpenCV
PIL (Pillow)
NumPy
scikit-learn
pdf2image
Gradio
 
 
