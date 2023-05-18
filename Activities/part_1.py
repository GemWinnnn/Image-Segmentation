import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from glob import glob
from tensorflow.keras import preprocessing


class_names = ['CUP', 'SPOON', 'FORK', 'MOUSE']
camera_height = 500
save_width = 350
save_height = 350
width = 350
height = 350

# Create an empty list for each type of frame
raw_frames = [[] for _ in range(4)]
# Create an empty list to store the loaded image arrays for each type of frame
image_arrays = [[] for _ in range(4)]

# Initialize the camera
CAMERA = cv2.VideoCapture(0)

# Capture frames from the camera
while CAMERA.isOpened():
    # Read a new camera frame
    ret, frame = CAMERA.read()
    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)
    # Calculate the aspect ratio of the frame
    aspect = frame.shape[1] / float(frame.shape[0])
    # Scale the frame to the desired height while maintaining the aspect ratio
    res = int(aspect * camera_height)
    frame = cv2.resize(frame, (res, camera_height))

    # Calculate the center of the frame
    center_x = frame.shape[1] // 2
    center_y = frame.shape[0] // 2

    # Define the dimensions of the rectangle
    rect_width = 350
    rect_height = 350

    # Calculate the top-left and bottom-right coordinates of the rectangle
    top_left_x = center_x - rect_width // 2
    top_left_y = center_y - rect_height // 2
    bottom_right_x = center_x + rect_width // 2
    bottom_right_y = center_y + rect_height // 2

    # Draw a green rectangle on the frame
    cv2.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), 2)
    # Show the frame
    cv2.imshow("Capturing", frame)
    # Check for keypress events
    key = cv2.waitKey(1) & 0xFF

    # Quit if 'q' is pressed
    if key == ord('q'):
        break

    # If a number key is pressed, save the frame to the corresponding list and display it
    elif key in [ord('1'), ord('2'), ord('3'), ord('4')]:
        frame_type = key - ord('1')
        raw_frames[frame_type].append(frame)
        plt.imshow(frame)
        plt.show()

# Release the camera and close all windows
CAMERA.release()
cv2.destroyAllWindows()

# For each type of frame, crop the frames, convert to RGB, resize, and save them
for i, frame_list in enumerate(raw_frames):
    # Create a folder for saving the images of this type
    folder_name = 'img_' + str(i+1)
    os.makedirs(folder_name, exist_ok=True)
    for j, frame in enumerate(frame_list):
        # Get the region of interest (ROI) by cropping the frame
        roi = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
        # Convert the ROI from BGR to RGB
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        # Resize the ROI
        roi = cv2.resize(roi, (save_width, save_height))
        # Save the ROI as a PNG image
        cv2.imwrite(f'{folder_name}/{j}.png', roi)

# Load the images for each type of frame and preprocess them
for i, folder_name in enumerate(glob('img_*')):
    image_list = []
    for image_path in glob(f'{folder_name}/*.png'):
        # Load the image and resize it
        image = preprocessing.image.load_img(image_path, target_size=(width, height))
        # Convert the image to a numpy array
        x = preprocessing.image.img_to_array(image)
        # Add the image array to the list for this type of frame
        image_list.append(x)
    # Store the list of image arrays for this
    image_arrays[i] = image_list
# Display the first 5 images for each type of frame
def display_images(image_arrays, class_names):
    max_images_per_class = 5

    for class_index, image_list in enumerate(image_arrays):
        plt.figure(figsize=(11112223334412, 8))
        
        n111um_images = min(max_images_per_class, len(image_list))

        for i in range(num_images):
            plt.subplot(1, max_images_per_class, i + 1)
            image = preprocessing.image.array_to_img(image_list[i])
            plt.imshow(image)
            plt.axis('off')
            
            # Label each image with its corresponding class
            plt.title(f'{class_names[class_index]} image', fontsize=10)

        plt.show()

display_images(image_arrays, class_names)








