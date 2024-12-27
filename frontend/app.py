import streamlit as st
import requests
import base64
from PIL import Image
import io

st.set_page_config(page_title="Pig Detection System", layout="wide")

st.title("🐷 Pig Detection System")

st.write("""
Upload an image to detect and count pigs. The system will analyze the image and 
show the detected pigs with bounding boxes and confidence scores.
""")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display original image
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
    
    # Make prediction
    files = {"file": uploaded_file.getvalue()}
    response = requests.post("http://localhost:8000/detect", files=files)
    
    if response.status_code == 200:
        result = response.json()
        
        with col2:
            st.subheader("Detection Result")
            # Display the result image
            img_data = base64.b64decode(result["image"])
            result_image = Image.open(io.BytesIO(img_data))
            st.image(result_image, use_column_width=True)
        
        # Display detection details
        st.subheader("Detection Details")
        st.write(f"Total pigs detected: {result['count']}")
        
        if result['count'] > 0:
            st.write("Individual detections:")
            for i, det in enumerate(result['detections'], 1):
                st.write(f"Pig {i}: Confidence: {det['confidence']:.2f}")
    else:
        st.error("Error processing the image. Please try again.")
