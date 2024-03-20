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


class FaceRecognitionManager:
    def __init__(self, database_folder):
        self.database_folder = database_folder
        self.encoded_faces_dir = os.path.abspath(os.path.join(self.database_folder, '..', 'Encoded_Faces'))
        self.excluded_dir = os.path.abspath(os.path.join(self.database_folder, '..', 'excluded_images'))
        self.model_dir = os.path.abspath(os.path.join(self.database_folder, '..', 'SVM_Models'))
        os.makedirs(self.encoded_faces_dir, exist_ok=True)
        os.makedirs(self.excluded_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

    def encode_faces_from_database(self):
        face_encodings_data = []
        face_labels = []

        person_folders = [os.path.join(self.database_folder, person) for person in os.listdir(self.database_folder) if os.path.isdir(os.path.join(self.database_folder, person))]
        
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
        svm_classifier = SVC(kernel='linear', probability=True)
        svm_classifier.fit(face_encodings_data, face_labels_num)

        # Save the trained model and label encoder
        self._save_model(svm_classifier, le)

    def predict_faces_from_database(self):
        # Load the trained SVM model and label encoder
        svm_classifier, le = self._load_model()
        if svm_classifier is None or le is None:
            print("No trained SVM model found. Please train the model first.")
            return

        person_folders = [os.path.join(self.database_folder, person) for person in os.listdir(self.database_folder) if os.path.isdir(os.path.join(self.database_folder, person))]
        total_images = sum([len(os.listdir(os.path.join(person_folder, 'test'))) for person_folder in person_folders if os.path.exists(os.path.join(person_folder, 'test'))])

        all_actual_labels = []
        all_predicted_labels = []

        with tqdm(total=total_images, desc="Predicting") as pbar:
            for person_folder in person_folders:
                actual_label = os.path.basename(person_folder)
                test_folder = os.path.join(person_folder, 'test')

                if os.path.exists(test_folder):
                    image_files = [os.path.join(test_folder, image_file) for image_file in os.listdir(test_folder) if image_file.lower().endswith(('.png', '.jpg', '.jpeg'))]

                    for image_file in image_files:
                        image = face_recognition.load_image_file(image_file)
                        encodings = face_recognition.face_encodings(image)

                        for face_encoding in encodings:
                            probabilities = svm_classifier.predict_proba([face_encoding])[0]
                            predicted_label_index = np.argmax(probabilities)
                            predicted_label = le.inverse_transform([predicted_label_index])[0]
                            prediction_percentage = probabilities[predicted_label_index] * 100

                            all_actual_labels.append(actual_label)
                            all_predicted_labels.append(predicted_label)

                            print(f"Actual: {actual_label}, Predicted: {predicted_label} ({prediction_percentage:.2f}%)")
                        pbar.update(1)

        # Generate and print the confusion matrix
        cm = confusion_matrix(all_actual_labels, all_predicted_labels, labels=le.classes_)

        # Plotting the confusion matrix
        plt.figure(figsize=(10, 7))
        sns.set(font_scale=1.2)  # for label size
        sns.heatmap(cm, annot=True, annot_kws={"size": 10}, fmt='g', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.show()

        # Calculate and print the accuracy
        accuracy = accuracy_score(all_actual_labels, all_predicted_labels)
        print(f"\nAccuracy: {accuracy*100:.2f}%")

    def _save_model(self, model, label_encoder):
        model_path = os.path.join(self.model_dir, f'svm_face_recognition_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump((model, label_encoder), f)
        print(f"Model saved at: {model_path}")

    def _load_model(self):
        try:
            model_files = [f for f in os.listdir(self.model_dir) if f.endswith('.pkl')]
            model_files.sort(reverse=True)
            latest_model_file = os.path.join(self.model_dir, model_files[0])
            with open(latest_model_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading model: {e}")
            return None, None

    def filter_valid_images(self):
        os.makedirs(self.excluded_dir, exist_ok=True)

        person_folders = [os.path.join(self.database_folder, person) for person in os.listdir(self.database_folder) if os.path.isdir(os.path.join(self.database_folder, person))]
        
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
                                new_file_path = os.path.join(self.excluded_dir, new_file_name)
                                shutil.move(image_file, new_file_path)
                                print(f"Moved: {image_file} -> {new_file_path}")
                            
                            pbar.update(1)
