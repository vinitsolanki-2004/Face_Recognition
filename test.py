import cv2
import face_recognition
import argparse
import pickle


def main(video_source, model_pickle):
    # Load the SVM model and label encoder from the pickle file
    with open(model_pickle, 'rb') as f:
        svm_classifier, label_encoder = pickle.load(f)

    # Open video capture
    if video_source == "0":
        video_source = 0
    cap = cv2.VideoCapture(video_source)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame from BGR (OpenCV default) to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Find all face locations and face encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_frame, model='hog')
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Predict the face label using the SVM model
            prediction = svm_classifier.predict([face_encoding])
            predicted_label_num = prediction[0]
            name = label_encoder.inverse_transform([predicted_label_num])[0]

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Draw the label below the face
            cv2.putText(frame, name, (left, bottom + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Face Recognition on Video Stream Using SVM')
    parser.add_argument('--video', type=str, required=True, help="0 for camera or file path for video")
    parser.add_argument('--model', type=str, required=True, help="Path to the pickle file with the SVM model and label encoder.")
    args = parser.parse_args()
    
    main(args.video, args.model)
