"""
Streamlit frontend for pig detection and tracking system.

This module implements a user-friendly interface using Streamlit to interact with the
FastAPI backend for pig detection and tracking visualization.
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
import io
import time
from typing import List, Dict, Optional, Union
import logging
import cv2
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Configuration
API_URL = "http://localhost:8000"

class PigDetectionUI:
    """
    Main UI class for the Pig Detection System.
    
    This class handles all frontend logic and API communication.
    """
    
    def __init__(self):
        """Initialize the UI components and session state."""
        st.set_page_config(
            page_title="Pig Detection System",
            page_icon="🐷",
            layout="wide"
        )
        
        # Initialize session state
        if "processing_history" not in st.session_state:
            st.session_state.processing_history = []
        if "current_result" not in st.session_state:
            st.session_state.current_result = None
            
    def check_api_health(self) -> bool:
        """Check if the API is accessible and healthy."""
        try:
            response = requests.get(f"{API_URL}/api/v1/health")
            if response.status_code == 200:
                return True
            return False
        except requests.RequestException:
            return False
            
    def setup_sidebar(self) -> str:
        """
        Configure the sidebar with control options.
        
        Returns:
            str: Selected processing mode
        """
        st.sidebar.title("Control Panel")
        
        # API Status
        api_status = "🟢 Online" if self.check_api_health() else "🔴 Offline"
        st.sidebar.markdown(f"### API Status: {api_status}")
        
        # Mode selection
        mode = st.sidebar.radio(
            "Select Mode",
            ["Single Image", "Video Processing", "Batch Processing"]
        )
        
        # Display processing history
        if st.sidebar.checkbox("Show Processing History"):
            self.display_history()
            
        return mode
        
    def display_history(self):
        """Display processing history in the sidebar."""
        st.sidebar.subheader("Processing History")
        for entry in st.session_state.processing_history[-5:]:  # Show last 5 entries
            st.sidebar.text(
                f"{entry['timestamp']}: {entry['type']} - {entry['count']} detections"
            )
            
    def process_image(self, image_file: Union[str, Path, io.BytesIO]) -> Optional[dict]:
        """
        Process a single image through the API.
        
        Args:
            image_file: Image file to process
            
        Returns:
            Optional[dict]: API response data if successful
        """
        try:
            files = {"file": image_file}
            with st.spinner("Processing image..."):
                response = requests.post(
                    f"{API_URL}/api/v1/detect/image",
                    files=files
                )
                
            if response.status_code == 200:
                result = response.json()
                # Update history
                st.session_state.processing_history.append({
                    "timestamp": time.strftime("%H:%M:%S"),
                    "type": "Image",
                    "count": result["detection_count"]
                })
                return result
            else:
                st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
                return None
                
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            return None
            
    def process_video(self, video_file: Union[str, Path, io.BytesIO]) -> Optional[dict]:
        """
        Process a video through the API.
        
        Args:
            video_file: Video file to process
            
        Returns:
            Optional[dict]: API response data if successful
        """
        try:
            files = {"file": video_file}
            with st.spinner("Processing video... This may take a while."):
                response = requests.post(
                    f"{API_URL}/api/v1/detect/video",
                    files=files
                )
                
            if response.status_code == 200:
                result = response.json()
                # Update history
                st.session_state.processing_history.append({
                    "timestamp": time.strftime("%H:%M:%S"),
                    "type": "Video",
                    "count": result["detection_count"]
                })
                return result
            else:
                st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
                return None
                
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
            return None
            
    def display_results(self, result: dict):
        """
        Display processing results including metrics and visualizations.
        
        Args:
            result: Processing result data from API
        """
        if not result:
            return
            
        # Display metrics in columns
        col1, col2, col3 = st.columns(3)
        col1.metric("Processing Time", f"{result['processing_time']:.2f}s")
        col2.metric("Detections", result["detection_count"])
        col3.metric("Status", result["status"])
        
        # Display result image/video
        if result.get("result_path"):
            try:
                result_url = f"{API_URL}/api/v1/results/{Path(result['result_path']).name}"
                
                if result["result_path"].endswith((".jpg", ".jpeg", ".png")):
                    st.image(result_url, caption="Detection Result", use_column_width=True)
                elif result["result_path"].endswith(".mp4"):
                    # Use requests to fetch the video data
                    response = requests.get(result_url, stream=True)
                    if response.status_code == 200:
                        st.video(response.content, format="video/mp4")
                    else:
                        st.error(f"Failed to fetch video: {response.status_code}")
                    
            except Exception as e:
                st.error(f"Error displaying result: {str(e)}")
                
        # Display tracked objects if available
        if result.get("tracked_objects"):
            st.subheader("Detected Objects")
            for track_id, obj in result["tracked_objects"].items():
                st.text(f"Pig #{track_id}: Confidence: {obj['confidence']:.2f}")
    
    def handle_single_image(self):
        """Handle single image processing mode."""
        st.header("Single Image Detection")
        
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png"],
            help="Upload an image for pig detection"
        )
        
        if uploaded_file:
            # Display original image
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Image", use_column_width=True)
            
            if st.button("Process Image"):
                result = self.process_image(uploaded_file)
                if result:
                    st.session_state.current_result = result
                    
        # Display current result if available
        if st.session_state.current_result:
            self.display_results(st.session_state.current_result)
            
    def handle_video_processing(self):
        """Handle video processing mode."""
        st.header("Video Processing")
        
        uploaded_file = st.file_uploader(
            "Choose a video",
            type=["mp4", "avi", "mov"],
            help="Upload a video for pig detection and tracking"
        )
        
        if uploaded_file:
            if st.button("Process Video"):
                result = self.process_video(uploaded_file)
                if result:
                    st.session_state.current_result = result
                    
        # Display current result if available
        if st.session_state.current_result:
            self.display_results(st.session_state.current_result)
            
    def handle_batch_processing(self):
        """Handle batch processing mode."""
        st.header("Batch Processing")
        
        uploaded_files = st.file_uploader(
            "Choose multiple images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            help="Upload multiple images for batch processing"
        )
        
        if uploaded_files:
            if st.button("Process Batch"):
                results = []
                progress_bar = st.progress(0)
                for i, file in enumerate(uploaded_files):
                    result = self.process_image(file)
                    if result:
                        results.append(result)
                    progress_bar.progress((i + 1) / len(uploaded_files))
                    
                if results:
                    st.success(f"Processed {len(results)} files successfully!")
                    # Display summary
                    total_detections = sum(r["detection_count"] for r in results)
                    avg_time = sum(r["processing_time"] for r in results) / len(results)
                    st.metric("Total Detections", total_detections)
                    st.metric("Average Processing Time", f"{avg_time:.2f}s")
    
    def run(self):
        """Main application loop."""
        st.title("🐷 Pig Detection System")
        
        mode = self.setup_sidebar()
        
        if mode == "Single Image":
            self.handle_single_image()
        elif mode == "Video Processing":
            self.handle_video_processing()
        else:  # Batch Processing
            self.handle_batch_processing()

if __name__ == "__main__":
    app = PigDetectionUI()
    app.run()
