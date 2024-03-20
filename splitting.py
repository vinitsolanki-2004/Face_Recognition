import os
import shutil
from sklearn.model_selection import train_test_split

def split_and_move_images(source_dir):
    # Get the parent directory of the source directory
    parent_dir = os.path.abspath(os.path.join(source_dir, os.pardir))

    # Create train and test directories in the parent directory
    train_dir = os.path.join(parent_dir, 'train')
    test_dir = os.path.join(parent_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Get list of images in the source directory
    images_list = os.listdir(source_dir)

    # Split images into train and test sets
    train_images, test_images = train_test_split(images_list, test_size=0.25, random_state=42)

    # Move train images to train directory
    for img in train_images:
        shutil.move(os.path.join(source_dir, img), os.path.join(train_dir, img))

    # Move test images to test directory
    for img in test_images:
        shutil.move(os.path.join(source_dir, img), os.path.join(test_dir, img))

    shutil.move(train_dir, os.path.join(source_dir, os.path.basename(train_dir)))

    shutil.move(test_dir, os.path.join(source_dir, os.path.basename(test_dir)))

    print(f"{len(train_images)} images moved to {train_dir}")
    print(f"{len(test_images)} images moved to {test_dir}")
