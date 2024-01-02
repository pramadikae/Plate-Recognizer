import os
from io import BytesIO

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image


class LicensePlateDetection:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = self.load_model()
        self.classes = self.model.names
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        st.write("Using Device:", self.device)

    def load_model(self):
        """
        Load YOLOv5 model
        """
        if self.model_path:
            model = torch.hub.load(
                "ultralytics/yolov5", "custom", path=model_path, force_reload=True
            )
        return model

    def score_image(self, image):
        """
        Calculate scores for a single frame using YOLOv5 model
        """
        self.model.to(self.device)
        image = [image]
        result = self.model(image)
        labels, coordinates = result.xyxyn[0][:, -1], result.xyxyn[0][:, :-1]
        return labels, coordinates

    def class_to_label(self, x):
        return self.classes[int(x)]

    def draw_boxes(self, result, image, image_path):
        """
        Draw bounding boxes on the image and save the detected image
        """
        detected_images = []
        labels, coordinates = result
        n = len(labels)
        x_shape, y_shape = image.shape[1], image.shape[0]

        for i in range(n):
            row = coordinates[i]
            if row[4] >= 0.3:
                x1, y1, x2, y2 = (
                    int(row[0] * x_shape),
                    int(row[1] * y_shape),
                    int(row[2] * x_shape),
                    int(row[3] * y_shape),
                )
                bgr = (0, 255, 0)

                confidence_score = row[4]
                text_score = f"{confidence_score:.2f}%"

                # Determine the position of the confidence score text
                x_text_score = x1 + 550
                y_text_score = y1 - 35

                cv2.rectangle(image, (x1, y1), (x2, y2), bgr, 8)
                cv2.putText(
                    image,
                    self.class_to_label(labels[i]),
                    (x1, y1 - 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    3,
                    bgr,
                    7,
                )
                cv2.putText(
                    image,
                    text_score,
                    (x_text_score, y_text_score),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    3,
                    bgr,
                    7,
                )

                # Crop the detected region without bounding box
                cropped_region = image[y1:y2, x1:x2]
                detected_images.append(cropped_region)

                # Save the detected image
                detected_image_path = os.path.join(
                    "detected_images", f"detected_image_{i}.jpg"
                )
                cv2.imwrite(detected_image_path, image)
                self.save_detection_info(
                    image_path, i, x1, y1, x2, y2, confidence_score
                )

        return image, detected_images

    def save_detection_info(self, image_path, index, x1, y1, x2, y2, confidence_score):
        """
        Save bounding box information to a text file
        """
        info_file_path = os.path.join("detected_images", f"detection_info_{index}.txt")
        with open(info_file_path, "w") as f:
            f.write(f"Image Path: {image_path}\n")
            f.write(f"Bounding Box Coordinates: {x1}, {y1}, {x2}, {y2}\n")
            f.write(f"Confidence Score: {confidence_score:.2f}%\n")

    def process_image(self, image, image_path):
        """
        Process a single image
        """
        # Create the "detected_images" folder if it doesn't exist
        if not os.path.exists("detected_images"):
            os.makedirs("detected_images")

        result, detected_images = self.draw_boxes(
            self.score_image(image), image, image_path
        )

        # Save the detected image
        result_image_path = os.path.join("detected_images", "result_image.jpg")
        cv2.imwrite(result_image_path, result)

        # Display the detected image
        result_image = Image.fromarray(result[..., ::-1])  # Convert BGR to RGB
        st.image(result_image, channels="RGB", caption="Detected Image")

        # Display and save cropped images
        for i, cropped_image in enumerate(detected_images):
            st.image(cropped_image, channels="BGR", caption=f"Cropped Image {i}")
            cropped_image_path = os.path.join(
                "detected_images", f"cropped_image_{i}.jpg"
            )
            cv2.imwrite(cropped_image_path, cropped_image)
            st.write(f"Cropped image {i} saved at {cropped_image_path}")

        st.write(f"Detected image saved at {result_image_path}")


if __name__ == "__main__":
    st.title("License Plate Detection with Streamlit")

    model_path = st.text_input("Enter the model path:")
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if model_path and uploaded_file is not None:
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
        license_plate_detector = LicensePlateDetection(model_path)

        st.image(image, channels="BGR", caption="Original Image")
        license_plate_detector.process_image(image, uploaded_file.name)
