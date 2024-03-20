import os


def create_folder():
    # Specify the path to the parent directory
    parent_directory = '/Users/vinitsolanki/IdeaProjects/FaceRecognition/face_collection'

    name = str(input('Enter name of person : '))

    # Name of the new folder to be created
    new_folder = name

    # Path to the new folder
    new_folder_path = os.path.join(parent_directory, new_folder)

    # Check if the directory already exists
    if not os.path.exists(new_folder_path):
        # Create the directory
        os.makedirs(new_folder_path)
        print(f"Directory '{new_folder}' created successfully at '{parent_directory}'")
        return new_folder_path
    else:
        print(f"Directory '{new_folder}' already exists at '{parent_directory}'")
