import cv2
from create_new_folder import *
from splitting import *
from FaceRecognitionManager import *


def capture_image(new_folder_path):
    # Initialize the camera capture object with the cv2.VideoCapture class.
    count = 0
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Display the resulting frame
        cv2.imshow('frame', frame)

        # Wait for key press
        key = cv2.waitKey(1)

        # if the 'c' key is pressed, capture the frame
        if key == ord('c'):
            img_name = os.path.join(new_folder_path, f'captured_frame_{count}.jpg')
            cv2.imwrite(img_name, frame)
            print(f"Image captured and saved as '{img_name}'")
            count += 1

        # if the 'q' key is pressed, break from the loop
        if key == ord('q'):
            print("Exiting...")
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    new_folder_path = create_folder()
    capture_image(new_folder_path)
    split_and_move_images(new_folder_path)
