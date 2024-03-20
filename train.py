from FaceRecognitionManager import FaceRecognitionManager
import argparse
import os
import pickle
from tqdm import tqdm
import face_recognition
from datetime import datetime
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
import shutil

# def update_model():

def encode_faces_from_database():
    face_encodings_data = []
    face_labels = []

    person_folders = '/Users/vinitsolanki/IdeaProjects/FaceRecognition/additional_images'

    total_images = sum([len(os.listdir(os.path.join(person_folder, 'train'))) for person_folder in person_folders if os.path.exists(os.path.join(person_folder, 'train'))])

    with tqdm(total=total_images, desc="Encoding") as pbar:
        for person_folder in person_folders:
            label = os.path.basename(person_folder)
            train_folder = os.path.join(person_folder, 'train')

            if os.path.exists(train_folder):
                image_files = [os.path.join(train_folder, image_file) for image_file in os.listdir(train_folder) if image_file.lower().endswith(('.png', '.jpg', '.jpeg'))]

                for image_file in image_files:
                    image = face_recognition.load_image_file(image_file)
                    encodings = face_recognition.face_encodings(image)

                    for encoding in encodings:
                        face_encodings_data.append(encoding)
                        face_labels.append(label)
                    pbar.update(1)

    # Convert labels to unique numerical values
    le = LabelEncoder()
    face_labels_num = le.fit_transform(face_labels)

    # Train SVM classifier
    svm_classifier, le = _load_model()
    if svm_classifier is None or le is None:
        print("No trained SVM model found. Please train the model first.")
        return
    svm_classifier.fit(face_encodings_data, face_labels_num)

    # Save the trained model and label encoder
    # self._save_model(svm_classifier, le)


def _load_model():
    try:
        latest_model_file = '/Users/vinitsolanki/IdeaProjects/FaceRecognition/SVM_Models/svm_face_recognition_20240314_154749.pkl_copy'
        with open(latest_model_file, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None


def filter_valid_images():
    excluded_dir = '/Users/vinitsolanki/IdeaProjects/FaceRecognition/excluded_images'

    person_folders = '/Users/vinitsolanki/IdeaProjects/FaceRecognition/additional_images'

    total_images = sum([len(os.listdir(os.path.join(person_folder, sub_folder))) for person_folder in person_folders for sub_folder in ['train', 'test'] if os.path.exists(os.path.join(person_folder, sub_folder))])

    with tqdm(total=total_images, desc="Filtering") as pbar:
        for person_folder in person_folders:
            person_name = os.path.basename(person_folder)

            for sub_folder in ['train', 'test']:
                folder_path = os.path.join(person_folder, sub_folder)

                if os.path.exists(folder_path):
                    image_files = [os.path.join(folder_path, image_file) for image_file in os.listdir(folder_path) if image_file.lower().endswith(('.png', '.jpg', '.jpeg'))]

                    for image_file in image_files:
                        image = face_recognition.load_image_file(image_file)
                        face_count = len(face_recognition.face_locations(image))

                        if face_count != 1:
                            new_file_name = f"{person_name}_{sub_folder}_{os.path.basename(image_file)}"
                            new_file_path = os.path.join(excluded_dir, new_file_name)
                            shutil.move(image_file, new_file_path)
                            print(f"Moved: {image_file} -> {new_file_path}")

                        pbar.update(1)


if __name__ == "__main__":
    # manager = FaceRecognitionManager(database_path)
    filter_valid_images()
    # encode_faces_from_database()
    # predict_faces_from_database()

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Process and predict faces from a given database folder.')
#     parser.add_argument('--database', type=str, required=False, default = "face_collection", help='Path to the database folder containing person-named folders.')
#
#     args = parser.parse_args()
#
#     main(args.database)
