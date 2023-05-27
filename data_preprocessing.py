import cv2
import numpy as np
import os
import random
import shutil

def make_dataset(dir,output_dir):
    for rootdir,dirs,files in os.walk(dir):
        for subdir in dirs:
            print(subdir)
            for files1 in os.listdir(os.path.join(rootdir,subdir)):
                
                image_path = os.path.join(os.path.join(rootdir,subdir),files1)
                img = None
                try:
                    # Load the image
                    img = cv2.imread(image_path)  
                except cv2.error as e:
                    # Rewrite the image for any exception if
                    cv2.imwrite(image_path, cv2.imread(image_path))
                    img = cv2.imread(image_path)
                if img is None:
                    print("invalid image :" + image_path)
                    continue
                
                # Get the original image size => height, weight, channels 
                h,w,c = img.shape
                
                # print(h,w,c)

                # Calculate the aspect ratio
                aspect_ratio = w/h
                
                # Determine the new size while maintaining the aspect ratio
                if aspect_ratio> 1:
                    new_w = 224
                    new_h = int(new_w / aspect_ratio)
                else:
                    new_h = 224
                    new_w = int(new_h * aspect_ratio)
                
                # Resize the image using the calculated dimensions
                resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                # Create a square canvas of target_size x target_size, here it is 224*224
                canvas = np.zeros((224, 224, 3), dtype=np.uint8)
                start_x = (224 - new_w) // 2
                start_y = (224 - new_h) // 2
                canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized_img
                
                # Get the filename and extension from the original image path
                filename = os.path.basename(image_path)
                name, extension = os.path.splitext(filename)

                # Create the output directory if it doesn't exist
                os.makedirs(output_dir, exist_ok=True)
                resized_data_dir_path = os.path.join(output_dir,subdir) 
                os.makedirs(resized_data_dir_path, exist_ok=True)

                # Save the resized image in the output directory
                output_path = os.path.join(resized_data_dir_path, f"{name}{extension}")
                cv2.imwrite(output_path, canvas)

def split_dataset(dir,output_dir):
    # Set the path to your dataset folder
    dataset_path = dir

    # Set the path to the output folders
    output_path = output_dir
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

def delete_directory(path):
    # Delete the directory and its contents recursively
    shutil.rmtree(path)

if __name__ == "__main__":
    
    # Set path for orignial dataset and for cleaned dataset path
    make_dataset(r'D:\Concordia\COMP 6721 AI\AI_PROJECT\training_set',r'D:\Concordia\COMP 6721 AI\AI_PROJECT\training_set_resized')
    
    # Set path for cleaned dataset and give path for train,valid and test dataset path
    split_dataset(r'D:\Concordia\COMP 6721 AI\AI_PROJECT\training_set_resized',r'D:\Concordia\COMP 6721 AI\AI_PROJECT')

    # Delete the training_set_resized directory to save space
    delete_directory(r'D:\Concordia\COMP 6721 AI\AI_PROJECT\training_set_resized')    