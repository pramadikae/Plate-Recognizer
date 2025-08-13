import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from src.plate_detection.detector import PlateDetector

if __name__ == "__main__":
    st.title("License Plate Detection with Streamlit")

    model_path = st.text_input("Enter the model path:")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if model_path and uploaded_file is not None:
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
        detector = PlateDetector(model_path)

        st.image(image, channels="BGR", caption="Original Image")
        
        # Process the image and get the detected images
        result_image, cropped_images = detector.process_image(image, uploaded_file.name)

        # Display the detected image with bounding boxes
        st.image(result_image, channels="BGR", caption="Detected Image")

        # Display and save cropped images
        if cropped_images:
            st.subheader("Cropped License Plates:")
            for i, cropped_img in enumerate(cropped_images):
                st.image(cropped_img, channels="BGR", caption=f"Cropped Image {i}")
                st.write(f"Cropped image {i} saved in detected_images/cropped_image_{i}.jpg")
        else:
            st.write("No license plates detected.")

        st.write("Detected image with bounding boxes saved in detected_images/detected_image_X.jpg")
