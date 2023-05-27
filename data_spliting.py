import os
import random
import shutil

# Set the path to your dataset folder
dataset_path = r"D:\Concordia\COMP 6721 AI\AI_PROJECT\training_set_resized"

# Set the path to the output folders
output_path = r"D:\Concordia\COMP 6721 AI\AI_PROJECT"
train_path = os.path.join(output_path, "train")
valid_path = os.path.join(output_path, "valid")
test_path = os.path.join(output_path, "test")

# Set the train-validation-test split ratio
train_ratio = 0.7
valid_ratio = 0.15
test_ratio = 0.15

# Create output directories if they don't exist
os.makedirs(train_path, exist_ok=True)
os.makedirs(valid_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

# Iterate through each bird species folder
for species_folder in os.listdir(dataset_path):
    print(species_folder)
    species_path = os.path.join(dataset_path, species_folder)
    if os.path.isdir(species_path):
        # Create species-specific output directories
        train_species_path = os.path.join(train_path, species_folder)
        valid_species_path = os.path.join(valid_path, species_folder)
        test_species_path = os.path.join(test_path, species_folder)
        os.makedirs(train_species_path, exist_ok=True)
        os.makedirs(valid_species_path, exist_ok=True)
        os.makedirs(test_species_path, exist_ok=True)
        
        # Get a list of image filenames in the species folder
        image_filenames = os.listdir(species_path)
        
        # Shuffle the image filenames
        random.shuffle(image_filenames)
        
        # Split the images into train, validation, and test sets
        train_size = int(len(image_filenames) * train_ratio)
        valid_size = int(len(image_filenames) * valid_ratio)
        train_filenames = image_filenames[:train_size]
        valid_filenames = image_filenames[train_size:train_size+valid_size]
        test_filenames = image_filenames[train_size+valid_size:]
        
        print("Train : {} , Valid : {} , Test : {} ".format(train_size,valid_size,((len(image_filenames))-(train_size+valid_size))))

        # Move train images to the train folder
        for filename in train_filenames:
            src = os.path.join(species_path, filename)
            dst = os.path.join(train_species_path, filename)
            shutil.copy(src, dst)
        
        # Move validation images to the validation folder
        for filename in valid_filenames:
            src = os.path.join(species_path, filename)
            dst = os.path.join(valid_species_path, filename)
            shutil.copy(src, dst)
        
        # Move test images to the test folder
        for filename in test_filenames:
            src = os.path.join(species_path, filename)
            dst = os.path.join(test_species_path, filename)
            shutil.copy(src, dst)
