import streamlit as st
import requests
from PIL import Image
import time
import os

# Set page title
st.title('Pig Detection and Counting App')

# File upload widget
file = st.file_uploader('Upload an image or video', type=['jpg', 'png', 'mp4'])

# Select type of file
file_type = st.selectbox('Select file type', ['Image', 'Video'])

# Button to trigger processing
process_button = st.button('Process')

# Area to display results
result_area = st.empty()

if process_button and file:
    if file_type == 'Image':
        # Process image
        files = {'file': file.read()}
        response = requests.post('http://localhost:8000/detect/image', files=files)
        if response.status_code == 200:
            data = response.json()
            st.success('Processing complete.')
            st.write(f'Total pigs detected: {data["total_count"]}')
            # Display image
            image = Image.open(file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
        else:
            st.error('An error occurred during processing.')
    elif file_type == 'Video':
        # Process video
        files = {'file': file.read()}
        response = requests.post('http://localhost:8000/detect/video', files=files)
        if response.status_code == 200:
            task_id = response.json()['task_id']
            st.info('Processing video. This may take some time.')
            # Poll for task status
            while True:
                status_response = requests.get(f'http://localhost:8000/task/{task_id}')
                status_data = status_response.json()
                if status_data['state'] == 'SUCCESS':
                    download_response = requests.get(f'http://localhost:8000/download/{task_id}')
                    if download_response.status_code == 200:
                        file_path = download_response.json()['file_path']
                        # Display video
                        st.video(open(file_path, 'rb').read())
                        st.success('Processing complete.')
                        break
                elif status_data['state'] == 'FAILURE':
                    st.error('An error occurred during processing.')
                    break
                time.sleep(2)
        else:
            st.error('An error occurred during processing.')