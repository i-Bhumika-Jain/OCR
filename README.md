### OCR Tool using PaddleOCR and Gradio

This repository provides a tool to perform Optical Character Recognition (OCR) on images and PDF files using the PaddleOCR library. The tool can process single images or multi-page PDF documents, cluster text lines using the DBSCAN algorithm, and visualize the OCR results.

**Key Features**

OCR on Images and PDFs: Supports extracting text from both images and PDF documents.
Text Clustering: Utilizes the DBSCAN algorithm to cluster text lines based on their y-coordinates.
Visualization: Draws bounding boxes around detected text and displays the annotated image.
Gradio Web Interface: Offers an intuitive web interface for easy interaction and quick results.

**Requirements**

Python 3.7 or higher ,
PaddleOCR ,
OpenCV ,
scikit-learn ,
numpy ,
pdf2image ,
PIL ,
Gradio 
