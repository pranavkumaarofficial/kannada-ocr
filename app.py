# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 23:18:33 2023

@author: Pranav
"""

import os
import pytesseract
import streamlit as st
from PIL import Image
import os
import cv2
import numpy as np
from skimage import io
from skimage.filters import threshold_otsu
from skimage.transform import rotate
from skimage.morphology import skeletonize, thin
from skimage import exposure

# Set the Tesseract executable path
tesseract_cmd = r'C:/Users/Pranav/Desktop/tesseract_data'

# Set the directory paths
upload_directory = 'C:/Users/Pranav/Desktop/Simple OCR/Source'
preprocess_directory = 'C:/Users/Pranav/Desktop/Simple OCR/preprocessed'
output_directory = 'C:/Users/Pranav/Desktop/Simple OCR/textfiles'

# Function to preprocess and save the images
def preprocess_images():
    st.write("Preprocessing images...")
    for filename in os.listdir(upload_directory):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(upload_directory, filename)
            image = cv2.imread(image_path, 0)

            # Apply additional preprocessing steps
            # Thresholding
            _, binary_image = cv2.threshold(image, 160, 255, cv2.THRESH_BINARY_INV)

            # Invert the image for better OCR results
            inverted_image = cv2.bitwise_not(binary_image)

            # Save the preprocessed image
            preprocessed_image_path = os.path.join('C:/Users/Pranav/Desktop/Simple OCR/preprocessed', filename)
            io.imsave(preprocessed_image_path, inverted_image.astype(np.uint8))
            #print(f"Preprocessing complete for: {filename}")
            
            
            #st.write(f"Preprocessing complete for: {filename}")

    st.success("Image preprocessing completed successfully!")


# Function to extract text from preprocessed images using OCR
def extract_text():
    tesseract_cmd = r'C:/Users/Pranav/Desktop/tesseract_data'
    st.write("Extracting text from preprocessed images...")
    
    for filename in os.listdir(preprocess_directory):
        if filename.endswith('.jpg') or filename.endswith('.png'): 
            # Load the preprocessed image
            image_path = os.path.join(preprocess_directory, filename)  # Use directory3 for Kannada OCR
            image = Image.open(image_path)
    
            # Perform Kannada OCR
            gray_image = image.convert('L')
            custom_config = r'--psm 6'  # Set page segmentation mode to treat the image as a single uniform block of text
            kannada_text = pytesseract.image_to_string(gray_image, lang='kan')
    
    
            # Create a new file with the same name as the image
            output_filename = os.path.splitext(filename)[0] + '.txt'
            output_filepath = os.path.join(output_directory, output_filename)  # Specify the directory to save the output file
    
            # Save the extracted OCR texts in the output file
            with open(output_filepath, 'w', encoding='utf-8') as f:
                #sf.write("Kannada OCR Text:\n")
                f.write(kannada_text)
            print(f"OCR texts extracted from {filename} saved in {output_filename}")
            st.write(f"OCR text extracted from {filename} and saved in {output_filename}")

    st.success("Text extraction completed successfully!")


# Streamlit app
def main():
    st.title("KANNADA OCR Tool")

    # Button to upload images

 
    if st.button("Upload Images", key="upload"):
        uploaded_files = st.file_uploader("Upload one or more images", accept_multiple_files=True)
        if uploaded_files:
            source_dir ='C:/Users/Pranav/Desktop/Simple OCR/Source'
            os.makedirs(source_dir, exist_ok=True)
            
            for uploaded_file in uploaded_files:
                file_path = os.path.join(source_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            
            st.success("Source images uploaded successfully!")
        
   
    # Button to extract text
    if st.button("Extract Text"):
        preprocess_images()
        extract_text()


if __name__ == "__main__":
    main()
