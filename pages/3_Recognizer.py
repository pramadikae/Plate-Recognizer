import os

import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow import keras


class PlateRecognizer:
    def __init__(self, image_path, model_path):
        self.image_path = image_path
        self.model_path = model_path

    def load_image(self):
        image = cv2.imread(self.image_path)
        image = imutils.resize(image, width=350)
        return image

    def display_image(self, image, title):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image_rgb, caption=title, use_column_width=True)

    def preprocess_image(self, image):
        img_plate_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        (thresh, img_plate_bw) = cv2.threshold(
            img_plate_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        img_plate_bw = cv2.morphologyEx(img_plate_bw, cv2.MORPH_OPEN, kernel)
        return img_plate_bw

    def find_character_candidates(self, img_plate_bw):
        contours_plate, _ = cv2.findContours(
            img_plate_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        index_chars_candidate = []

        for index, contour_plate in enumerate(contours_plate):
            x_char, y_char, w_char, h_char = cv2.boundingRect(contour_plate)
            if 40 <= h_char <= 60 and w_char >= 10:
                index_chars_candidate.append(index)

        return contours_plate, index_chars_candidate

    def display_candidate_characters(self, image, contours, indices, title):
        img_plate_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        for char_index in indices:
            x_char, y_char, w_char, h_char = cv2.boundingRect(contours[char_index])
            cv2.rectangle(
                img_plate_rgb,
                (x_char, y_char),
                (x_char + w_char, y_char + h_char),
                (0, 255, 0),
                2,
            )

        self.display_image(img_plate_rgb, title)

    def calculate_scores(self, indices, contours):
        score_chars_candidate = np.zeros(len(indices))

        for counter_index_chars_candidate, chars_candidateA in enumerate(indices):
            xA, yA, wA, hA = cv2.boundingRect(contours[chars_candidateA])

            for chars_candidateB in indices:
                if chars_candidateA == chars_candidateB:
                    continue
                else:
                    xB, yB, wB, hB = cv2.boundingRect(contours[chars_candidateB])
                    y_difference = abs(yA - yB)

                    if y_difference < 15:
                        score_chars_candidate[counter_index_chars_candidate] += 1

        return score_chars_candidate

    def get_sorted_characters(self, indices, contours, scores):
        index_chars = []

        for counter_index_chars_candidate, score in enumerate(scores):
            if score == max(scores):
                index_chars.append(indices[counter_index_chars_candidate])

        x_coors = [cv2.boundingRect(contours[char])[0] for char in index_chars]
        x_coors = sorted(x_coors)

        index_chars_sorted = []
        for x_coor in x_coors:
            for char in index_chars:
                x, _, _, _ = cv2.boundingRect(contours[char])
                if x_coors[x_coors.index(x_coor)] == x:
                    index_chars_sorted.append(char)

        return index_chars_sorted

    def recognize_characters(self, indices_sorted, contours, img_plate_bw):
        img_height = 40
        img_width = 40
        class_names = [
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "J",
            "K",
            "L",
            "M",
            "N",
            "O",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "U",
            "V",
            "W",
            "X",
            "Y",
            "Z",
        ]

        model = keras.models.load_model(self.model_path)

        num_plate = []
        for char_sorted in indices_sorted:
            x, y, w, h = cv2.boundingRect(contours[char_sorted])
            char_crop = cv2.cvtColor(
                img_plate_bw[y : y + h, x : x + w], cv2.COLOR_GRAY2BGR
            )
            char_crop = cv2.resize(char_crop, (img_width, img_height))
            img_array = keras.preprocessing.image.img_to_array(char_crop)
            img_array = tf.expand_dims(img_array, 0)
            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0])
            num_plate.append(class_names[np.argmax(score)])

        return num_plate

    def display_final_result(self, indices_sorted, contours, img_plate_rgb, num_plate):
        for char_sorted in indices_sorted:
            x, y, w, h = cv2.boundingRect(contours[char_sorted])
            cv2.rectangle(img_plate_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                img_plate_rgb,
                str(indices_sorted.index(char_sorted)),
                (x, y + h + 25),
                cv2.FONT_ITALIC,
                1.0,
                (0, 0, 255),
                2,
            )

        plate_number = "".join(num_plate)
        cv2.putText(
            img_plate_rgb,
            plate_number,
            (0, y + h + 50),
            cv2.FONT_ITALIC,
            1.0,
            (0, 0, 255),
            2,
        )

        self.display_image(img_plate_rgb, "Final Result")


def main():
    st.title("Character Recognition with Streamlit")

    image_path = "detected_images/cropped_image_0.jpg"
    model_path = "C:/Users/dikae/Documents/GitHub/Plate-Recognizer/my_model"  # Update with the correct path

    plate_recognizer = PlateRecognizer(image_path, model_path)

    # Original Image
    original_image = plate_recognizer.load_image()
    plate_recognizer.display_image(original_image, "1. Original Image")

    # Grayscale Image
    img_plate_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    plate_recognizer.display_image(img_plate_gray, "2. Grayscale Image")

    # Otsu Filter
    _, img_plate_otsu = cv2.threshold(
        img_plate_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    plate_recognizer.display_image(img_plate_otsu, "3. Otsu Filter")

    # Contours Filter
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    img_plate_contours = cv2.morphologyEx(img_plate_otsu, cv2.MORPH_OPEN, kernel)
    plate_recognizer.display_image(img_plate_contours, "4. Contours Filter")

    # Character Candidates
    contours, indices = plate_recognizer.find_character_candidates(img_plate_contours)
    plate_recognizer.display_candidate_characters(
        img_plate_contours, contours, indices, "5. Candidate Characters"
    )

    # Sorting Characters
    scores = plate_recognizer.calculate_scores(indices, contours)
    indices_sorted = plate_recognizer.get_sorted_characters(indices, contours, scores)
    num_plate = plate_recognizer.recognize_characters(
        indices_sorted, contours, img_plate_contours
    )
    plate_recognizer.display_final_result(
        indices_sorted, contours, original_image, num_plate
    )


if __name__ == "__main__":
    main()
