import cv2
import numpy as np
import os
def make_dataset(dir,output_dir):
    avg_h = 0
    avg_w = 0
    avg_c = 0
    image_count = 0
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
                image_count += 1
                
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

                avg_h += h
                avg_w += w
                avg_c += c
                
    print(avg_h/image_count)
    print(avg_w/image_count)
    print(avg_c/image_count)        


if __name__ == "__main__":
    make_dataset(r'D:\Concordia\COMP 6721 AI\AI_PROJECT\training_set',r'D:\Concordia\COMP 6721 AI\AI_PROJECT\training_set_resized')    