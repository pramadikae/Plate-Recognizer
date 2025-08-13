
import cv2
import imutils
import numpy as np
import tensorflow as tf
from tensorflow import keras

class CharacterRecognizer:
    def __init__(self, model_path):
        self.model_path = model_path
        self.img_height = 40
        self.img_width = 40
        self.class_names = [
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
            "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
            "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
            "U", "V", "W", "X", "Y", "Z"
        ]
        self.model = self._load_model()

    def _load_model(self):
        return keras.models.load_model(self.model_path)

    def preprocess_image(self, image):
        img_plate_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, img_plate_bw = cv2.threshold(
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
                    break # Add break to avoid duplicate characters if x-coordinates are the same
        return index_chars_sorted

    def recognize_characters(self, indices_sorted, contours, img_plate_bw):
        num_plate = []
        for char_sorted in indices_sorted:
            x, y, w, h = cv2.boundingRect(contours[char_sorted])
            char_crop = cv2.cvtColor(
                img_plate_bw[y : y + h, x : x + w], cv2.COLOR_GRAY2BGR
            )
            char_crop = cv2.resize(char_crop, (self.img_width, self.img_height))
            img_array = keras.preprocessing.image.img_to_array(char_crop)
            img_array = tf.expand_dims(img_array, 0)
            predictions = self.model.predict(img_array)
            score = tf.nn.softmax(predictions[0])
            num_plate.append(self.class_names[np.argmax(score)])
        return "".join(num_plate)

    def process_image(self, image_path):
        image = cv2.imread(image_path)
        image = imutils.resize(image, width=350)

        img_plate_bw = self.preprocess_image(image)
        contours, indices = self.find_character_candidates(img_plate_bw)

        if not indices:
            return "No characters detected."

        scores = self.calculate_scores(indices, contours)
        indices_sorted = self.get_sorted_characters(indices, contours, scores)
        
        plate_number = self.recognize_characters(indices_sorted, contours, img_plate_bw)
        return plate_number
