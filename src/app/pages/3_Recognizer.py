import streamlit as st
import cv2
import imutils
import numpy as np
from src.character_recognition.recognizer import CharacterRecognizer

def main():
    st.title("Character Recognition with Streamlit")

    image_path = "detected_images/cropped_image_0.jpg"
    model_path = "C:/Users/dikae/Documents/GitHub/Plate-Recognizer/my_model"  # Update with the correct path

    if not os.path.exists(image_path):
        st.error(f"Image not found at {image_path}. Please run the Detection page first.")
        return

    # Load the image for display
    original_image = cv2.imread(image_path)
    original_image_resized = imutils.resize(original_image, width=350)
    st.image(cv2.cvtColor(original_image_resized, cv2.COLOR_BGR2RGB), caption="Original Image", use_column_width=True)

    recognizer = CharacterRecognizer(model_path)
    plate_number = recognizer.process_image(image_path)

    st.subheader("Recognized Plate Number:")
    st.write(f"## {plate_number}")

if __name__ == "__main__":
    main()

