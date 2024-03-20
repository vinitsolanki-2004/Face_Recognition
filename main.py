import argparse
from FaceRecognitionManager import FaceRecognitionManager


def main(database_path):
    manager = FaceRecognitionManager(database_path)
    manager.filter_valid_images()
    manager.encode_faces_from_database()
    manager.predict_faces_from_database()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process and predict faces from a given database folder.')
    parser.add_argument('--database', type=str, required=False, default = "face_collection", help='Path to the database folder containing person-named folders.')
    
    args = parser.parse_args()
    
    main(args.database)
