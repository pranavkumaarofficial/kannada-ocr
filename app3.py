import os
import cv2
import numpy as np
from skimage import io
from skimage.filters import threshold_otsu
from skimage.transform import rotate
from skimage.morphology import skeletonize, thin
from skimage import exposure
from skimage.morphology import skeletonize
import numpy as np
from skimage import io
from PIL import Image
import tempfile
import streamlit as st

from skimage import io
from PIL import Image
import tempfile
import streamlit as st

from PIL import Image
import pytesseract
from langdetect import detect
import streamlit as st


'''

############################################################################################


# Set the Tesseract executable path
tesseract_cmd = r'C:/Users/Pranav/Desktop/tesseract_data'
# Set the directory paths
upload_directory = 'C:\\Users\\Pranav\\Desktop\\Sumukha\\Code\\Simple OCR\\Source'
preprocess_directory = 'C:\\Users\\Pranav\\Desktop\\Sumukha\\Code\\Simple OCR\\preprocessed'
output_directory = 'C:\\Users\\Pranav\\Desktop\\Sumukha\\Code\\Simple OCR\\textfiles'



############################################################################################


'''


############################################################################################


# Set the Tesseract executable path
tesseract_cmd = r'/tesseract_data'
# Set the directory paths
upload_directory = '\\Source'
preprocess_directory = '\\preprocessed'
output_directory = '\\textfiles'



############################################################################################





st.markdown(
    """
    <style>
    .container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 80vh;
        font-family: Arial, sans-serif;
    }
    .header {
        font-size: 32px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .button {
        padding: 12px 24px;
        font-size: 18px;
        font-weight: bold;
        color: #ffffff;
        background-color: #FF5722;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .button:hover {
        background-color: #E64A19;
    }
    .result-box {
        margin-top: 40px;
        padding: 20px;
        font-size: 18px;
        background-color: #F5F5F5;
        border-radius: 5px;
        white-space: pre-wrap;
    }
    </style>
    """,
    unsafe_allow_html=True
)





#for scanned documents 
def preprocess_images():
    st.write("Preprocessing images...")
    for filename in os.listdir(upload_directory):
        if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'): 

            image_path = os.path.join(upload_directory, filename)
            image = cv2.imread(image_path, 0)

            norm_img = np.zeros((image.shape[0], image.shape[1]))
            image = cv2.normalize(image, norm_img, 0, 255, cv2.NORM_MINMAX)

            color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            denoised_image = cv2.fastNlMeansDenoisingColored(color_image, None, 5, 5, 7, 15)
            image = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2GRAY)
            kernel = np.ones((1, 1), np.uint8)
            image = cv2.erode(image, kernel, iterations=1)
            sharpening_kernel = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]])
            
            image = cv2.filter2D(image, -1, sharpening_kernel)
            
            _, binary_image1 = cv2.threshold(image, 160, 255, cv2.THRESH_BINARY_INV) 
            inverted_image1 = cv2.bitwise_not(binary_image1)
            ii = cv2.threshold(inverted_image1, 160, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            preprocessed_image_path = os.path.join(preprocess_directory, filename)
            io.imsave(preprocessed_image_path, ii.astype(np.uint8))

    st.success("Image preprocessing completed successfully!")





#for non scanned documents
def preprocess_images1():
    st.write("Preprocessing images...")
    for filename in os.listdir(upload_directory):
        if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'): 

            image_path = os.path.join(upload_directory, filename)
            image = cv2.imread(image_path, 0)

            norm_img = np.zeros((image.shape[0], image.shape[1]))
            image = cv2.normalize(image, norm_img, 0, 255, cv2.NORM_MINMAX)

            color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            # Apply the denoising function to the color image
            denoised_image = cv2.fastNlMeansDenoisingColored(color_image, None, 5, 5, 7, 15)

            # If needed, convert the denoised image back to grayscale
            image = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2GRAY)

            kernel = np.ones((1, 1), np.uint8)
            image = cv2.erode(image, kernel, iterations=1)
            

            
            _, binary_image1 = cv2.threshold(image, 160, 255, cv2.THRESH_BINARY_INV)
            inverted_image1 = cv2.bitwise_not(binary_image1)
            ii = cv2.threshold(inverted_image1, 160, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            preprocessed_image_path = os.path.join(preprocess_directory, filename)
            io.imsave(preprocessed_image_path, ii.astype(np.uint8))
    st.success("Image preprocessing completed successfully!")




# Function to extract text from preprocessed images using OCR
def extract_text():
   
    st.write("Extracting text from preprocessed images...")
    
    for filename in os.listdir(preprocess_directory):
        if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'): 
            image_path = os.path.join(preprocess_directory, filename)
            image = Image.open(image_path)
    
            # Perform Kannada OCR
            gray_image = image.convert('L')
            custom_config = r'--psm 6'  # Set page segmentation mode to treat the image as a single uniform block of text
            kannada_text = pytesseract.image_to_string(gray_image, lang='kan')
            output_filename = os.path.splitext(filename)[0] + '.txt'
            output_filepath = os.path.join(output_directory, output_filename)
    
            # Save the extracted OCR texts in the output file
            with open(output_filepath, 'w', encoding='utf-8') as f:
                f.write(kannada_text)
            print(f"OCR texts extracted from {filename} saved in {output_filename}")
            st.write(f"OCR text extracted from {filename} and saved in {output_filename}")

    st.success("Text extraction completed successfully!")



# Streamlit app
def main():
    st.title("KANNADA OCR Tool")

    # Button to upload images
    
    uploaded_files = st.file_uploader("Upload your file here...", type=['png', 'jpeg', 'jpg'], accept_multiple_files=True)
    source_dir = upload_directory
    for uploaded_file in uploaded_files:
        file_path = os.path.join(source_dir, uploaded_file.name)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())



    if st.button("Extract text from scanned doc"):
        preprocess_images()
        extract_text()

    if st.button("Extract Text from image"):
        preprocess_images1()
        extract_text()






    footer = """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f5f5f5;
        color: #666666;
        text-align: center;
        padding: 8px 0;
        font-size: 15px;
        font-family: Times New Roman, sans-serif;
        font-family: Consolas, sans-serif;
        font-style: italic;
        
    }
    </style>
    <div class="footer">
        Application developed by Pranav Kumaar

    </div>
    """
    st.markdown(footer, unsafe_allow_html=True)
    
    
    
    
    
    st.markdown("</div>", unsafe_allow_html=True)
   

if __name__ == "__main__":
    main()
