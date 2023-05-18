import matplotlib.pyplot as plt 
import numpy as np 
import cv2 
from glob import glob
from keras import preprocessing
import os
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Flatten, Dense
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from PIL import Image
import seaborn as sns 
from sklearn.metrics import confusion_matrix

# %matplotlib inline

class_names = ['HEADPHONE','USBDRIVE','MOUSE','POWERBANK']

# Initialize the camera
CAMERA = cv2.VideoCapture(0)

camera_height = 500
save_width = 350
save_height = 350
width = 350
height = 350

# Create an empty list for each type of frame
raw_frames = [[] for _ in range(4)]

# Initialize empty lists to store images of each type
image_arrays = [[] for _ in range(4)]

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
    
    # show the frame
    cv2.imshow('Capturing', frame)
    
    # controls q = quit/ s = capturing
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

# camera
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

 
#Prepare Image to Tensor
X = [np.array(images) for images in image_arrays]

# Check the shape of the image arrays
for i, images in enumerate(X):
    print(f'X_type_{i+1} shape: {images.shape}')

# Concatenate all arrays along the first dimension
X = np.concatenate(X, axis=0)

# Scaling the data to be between 0 and 1
X = X / 255.0

print(f'X shape: {X.shape}')

from keras.utils import to_categorical

# Create labels for each type of frame
y = [np.full((len(images),), i) for i, images in enumerate(image_arrays)]
y = np.concatenate(y, axis=0)

# One-hot encode the labels
y = to_categorical(y, num_classes=len(class_names))

print(f'y shape: {y.shape}')

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import Adamax

# Model configuration
conv_1 = 16
conv_1_drop = 0.2
conv_2 = 32
conv_2_drop = 0.2
dense_1_n = 1024
dense_1_drop = 0.2
dense_2_n = 512
dense_2_drop = 0.2

#values you can adjust
lr = 0.001
epochs = 5
batch_size = 10
color_channels = 3

def build_model():
    model = Sequential()

    model.add(Conv2D(conv_1, (3,3), input_shape= (width, height, color_channels), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(conv_1_drop))

    model.add(Conv2D(conv_2, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(conv_2_drop))

    model.add(Flatten())
    model.add(Dense(dense_1_n, activation='relu'))
    model.add(Dropout(dense_1_drop))

    model.add(Dense(dense_2_n, activation='relu'))
    model.add(Dropout(dense_2_drop))

    model.add(Dense(len(class_names), activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adamax(lr=lr), metrics=['accuracy'])

    return model

# Build the model
model = build_model()

# Print the model summary
model.summary()

# Train the model
history = model.fit(X, y, validation_split=0.10, epochs=epochs, batch_size=batch_size)

print(history)

# Model evaluation

scores = model.evaluate(X, y, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()

plt.plot(history.history['loss']) 
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()

# Prediction

HEADPHONE   = 'img_1/10.png'
USBDRIVE = 'img_2/10.png'
MOUSE  = 'img_3/10.png'
POWERBANK = 'img_4/10.png'

imgs  = [HEADPHONE,USBDRIVE,MOUSE,POWERBANK]

predicted_classes = []

for i in range(len(imgs)):
    type_img = preprocessing.image.load_img(imgs[i], target_size=(width, height))
    plt.imshow(type_img)
    plt.show()

    type_x = preprocessing.image.img_to_array(type_img)
    type_x = np.expand_dims(type_x, axis=0)
    type_x = type_x / 255.0

    prediction = model.predict(type_x)
    index = np.argmax(prediction)
    print(class_names[index])
    predicted_classes.append(class_names[index])

cm = confusion_matrix(class_names, predicted_classes)
f = sns.heatmap(cm, xticklabels=class_names, yticklabels=class_names, annot=True)

plt.show()

type_1 = preprocessing.image.load_img('img_1/10.png', target_size=(width, height))
plt.imshow(type_1)
plt.show()

type_1_x = preprocessing.image.img_to_array(type_1)
type_1_x = np.expand_dims(type_1_x, axis=0)
type_1_x = type_1_x / 255.0

predictions = model.predict(type_1_x)
index = np.argmax(predictions)

print(class_names[index])

type_2 = preprocessing.image.load_img('img_2/10.png', target_size=(width, height))
plt.imshow(type_2)
plt.show()

type_2_x = preprocessing.image.img_to_array(type_2)
type_2_x = np.expand_dims(type_2_x, axis=0)
type_2_x = type_2_x / 255.0

predictions = model.predict(type_2_x)
index = np.argmax(predictions)
print(class_names[index])


# Live predictions using camera
# Live Predictions using camera
import cv2
import time

CAMERA = cv2.VideoCapture(0)
camera_height = 500

while(True):
    _, frame = CAMERA.read()

    # Flip
    frame = cv2.flip(frame, 1)

    # Rescale the images output
    aspect = frame.shape[1] / float(frame.shape[0])
    res = int(aspect * camera_height)
    frame = cv2.resize(frame, (res, camera_height))

    # Get roi
    roi = frame[75+2:425-2, 300+2:650-2]

    # Parse BRG to RGB
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    # Adjust alignment
    roi = cv2.resize(roi, (width, height))
    roi_x = np.expand_dims(roi, axis=0)

    # Normalize roi_x before predictions
    roi_x = roi_x / 255.0

    predictions = model.predict(roi_x)
    type_1_x, type_2_x, type_3_x, type_4_x = predictions[0]

    # The green rectangle
    cv2.rectangle(frame, (300, 75), (650, 425), (240, 100, 0), 2)

    # Predictions / Labels
    tipe_1_txt = '{} - {}%'.format(class_names[0], int(type_1_x*100))
    cv2.putText(frame, tipe_1_txt, (70, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)

    tipe_2_txt = '{} - {}%'.format(class_names[1], int(type_2_x*100))
    cv2.putText(frame, tipe_2_txt, (70, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)

    tipe_3_txt = '{} - {}%'.format(class_names[2], int(type_3_x*100))
    cv2.putText(frame, tipe_3_txt, (70, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)

    tipe_4_txt = '{} - {}%'.format(class_names[3], int(type_4_x*100))
    cv2.putText(frame, tipe_4_txt, (70, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)

    # Show frame
    cv2.imshow("Real time object detection", frame)

    # Controls: q = quit
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

CAMERA.release()
cv2.destroyAllWindows()

