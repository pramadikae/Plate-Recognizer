import os
import cv2
import numpy as np
import torch
from PIL import Image

class PlateDetector:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = self._load_model()
        self.classes = self.model.names
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using Device:", self.device)

    def _load_model(self):
        if self.model_path:
            model = torch.hub.load(
                "ultralytics/yolov5", "custom", path=self.model_path, force_reload=True
            )
        return model

    def _score_image(self, image):
        self.model.to(self.device)
        image = [image]
        result = self.model(image)
        labels, coordinates = result.xyxyn[0][:, -1], result.xyxyn[0][:, :-1]
        return labels, coordinates

    def _class_to_label(self, x):
        return self.classes[int(x)]

    def process_image(self, image, image_name="input_image.jpg"):
        detected_images = []
        labels, coordinates = self._score_image(image)
        n = len(labels)
        x_shape, y_shape = image.shape[1], image.shape[0]

        # Create the "detected_images" folder if it doesn't exist
        output_dir = "detected_images"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

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

                # Draw bounding box and text on a copy of the original image for display
                display_image = image.copy()
                cv2.rectangle(display_image, (x1, y1), (x2, y2), bgr, 8)
                cv2.putText(
                    display_image,
                    self._class_to_label(labels[i]),
                    (x1, y1 - 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    3,
                    bgr,
                    7,
                )
                cv2.putText(
                    display_image,
                    text_score,
                    (x1 + 550, y1 - 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    3,
                    bgr,
                    7,
                )

                # Crop the detected region without bounding box
                cropped_region = image[y1:y2, x1:x2]
                detected_images.append(cropped_region)

                # Save the detected image with bounding boxes
                detected_image_path = os.path.join(output_dir, f"detected_image_{i}.jpg")
                cv2.imwrite(detected_image_path, display_image)

                # Save cropped image
                cropped_image_path = os.path.join(output_dir, f"cropped_image_{i}.jpg")
                cv2.imwrite(cropped_image_path, cropped_region)

                # Save detection info
                info_file_path = os.path.join(output_dir, f"detection_info_{i}.txt")
                with open(info_file_path, "w") as f:
                    f.write(f"Image Name: {image_name}\n")
                    f.write(f"Bounding Box Coordinates: {x1}, {y1}, {x2}, {y2}\n")
                    f.write(f"Confidence Score: {confidence_score:.2f}%")

        return image, detected_images
