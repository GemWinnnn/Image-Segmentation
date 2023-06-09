#Libraries 
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
from keras.utils import to_categorical
from keras.applications import inception_v3
import seaborn as sns 
import tensorflow as tf
from keras.preprocessing import image
from keras.utils import to_categorical


class_names = ['HEADPHONE','HARDDRIVE','MOUSE','GLASSES']

# Initialize the camera
CAMERA = cv2.VideoCapture(1)

camera_height = 500
save_width = 200
save_height = 200
width = 200
height = 200

# Create an empty list for each type of frame
raw_frames_type_1 = []
raw_frames_type_2 = []
raw_frames_type_3 = []
raw_frames_type_4 = []

# Initialize empty lists to store images of each type
images_type_1 = []
images_type_2 = []
images_type_3 = []
images_type_4 = []

while CAMERA.isOpened():
    
    # Read a new camera frame
    ret, frame = CAMERA.read()

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Rescale the image output
    aspect = frame.shape[1] / float(frame.shape[0])
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
    
    if key == ord('q'):
        break
    elif key == ord('1'):
        # save the raw frames to frame
        raw_frames_type_1.append(frame)
        print("Captured type 1 frame.")
    elif key == ord('2'):
        raw_frames_type_2.append(frame)
        print("Captured type 2 frame.")
    elif key == ord('3'):
        raw_frames_type_3.append(frame)
        print("Captured type 3 frame.")
    elif key == ord('4'):
        raw_frames_type_4.append(frame)
        print("Captured type 4 frame.")

#Camera
CAMERA.release()
cv2.destroyAllWindows()

retval = os.getcwd()
print ("Current working directory %s" % retval)

print ('img1: ', len(raw_frames_type_1))
print ('img2: ', len(raw_frames_type_2))
print ('img3: ', len(raw_frames_type_3))
print ('img4: ', len(raw_frames_type_4))

#crop the images

for i, frame in enumerate(raw_frames_type_1):
    
    #get roi
    roi = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

    #parse brg to rgb
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    
    #resize to 224 x 224
    roi = cv2.resize(roi, (save_width, save_height))
    
    #save
    cv2.imwrite('img_1/{}.png'.format(i), cv2.cvtColor(roi, cv2.COLOR_RGB2BGR))
    
    plt.imshow(roi)
    plt.axis('off')
    plt.show()

for i, frame in enumerate(raw_frames_type_2):
    
    #get roi
    roi = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
    
    #parse brg to rgb
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    
    #resize to 224 x 224
    roi = cv2.resize(roi, (save_width, save_height))
    
    #save
    cv2.imwrite('img_2/{}.png'.format(i), cv2.cvtColor(roi, cv2.COLOR_RGB2BGR))
    
    plt.imshow(roi)
    plt.axis('off')
    plt.show()


for i, frame in enumerate(raw_frames_type_3):
    
    #get roi
    roi = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
    
    #parse brg to rgb
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    
    #resize to 224 x 224
    roi = cv2.resize(roi, (save_width, save_height))
    
    #save2
    cv2.imwrite('img_3/{}.png'.format(i), cv2.cvtColor(roi, cv2.COLOR_RGB2BGR))
    
    plt.imshow(roi)
    plt.axis('off')
    plt.show()

for i, frame in enumerate(raw_frames_type_4):
    
    #get roi
    roi = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

    #parse brg to rgb
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    
    #resize to 224 x 224
    roi = cv2.resize(roi, (save_width, save_height))
    
    #save
    cv2.imwrite('img_4/{}.png'.format(i), cv2.cvtColor(roi, cv2.COLOR_RGB2BGR))

    plt.imshow(roi)
    plt.axis('off')
    plt.show()


for image_path in glob('img_1/*.png*'):
    image = tf.keras.utils.load_img(image_path, target_size=(width, height))
    x = tf.keras.utils.img_to_array(image)
    
    images_type_1.append(x)

for image_path in glob('img_2/*.png*'):
    image = tf.keras.utils.load_img(image_path, target_size=(width, height))
    x =tf.keras.utils.img_to_array(image)
    
    images_type_2.append(x)
    
for image_path in glob('img_3/*.png*'):
    image = tf.keras.utils.load_img(image_path, target_size=(width, height))
    x = tf.keras.utils.img_to_array(image)
    
    images_type_3.append(x)

for image_path in glob('img_4/*.png*'):
    image = tf.keras.utils.load_img (image_path, target_size=(width, height))
    x = tf.keras.utils.img_to_array(image)
    
    images_type_4.append(x)

plt.figure(figsize=(12,8))
    
# Generate visualization for images_type_1
for i, x in enumerate(images_type_1[:5]):
    
    plt.subplot(1, 5, i+1)
    image = tf.keras.utils.array_to_img(x)
    plt.imshow(image)
    
    plt.axis('off')
    plt.title('{} image'.format(class_names[0]))

plt.show()
plt.figure(figsize=(12, 8))

# Generate visualization for images_type_2
for i, x in enumerate(images_type_2[:5]):
    
    plt.subplot(1, 5, i+1)
    image = tf.keras.utils.array_to_img(x)
    plt.imshow(image)
    
    plt.axis('off')
    plt.title('{} image'.format(class_names[1]))

plt.show()
plt.figure(figsize=(12, 8))

# Generate visualization for images_type_3
for i, x in enumerate(images_type_3[:5]):
        
    plt.subplot(1, 5, i+1)
    image = tf.keras.utils.array_to_img(x)
    plt.imshow(image)
    
    plt.axis('off')
    plt.title('{} image'.format(class_names[2]))

plt.show()
plt.figure(figsize=(12, 8))

# Generate visualization for images_type_4
for i, x in enumerate(images_type_4[:5]):
    
    plt.subplot(1, 5, i+1)
    image = tf.keras.utils.array_to_img(x)
    plt.imshow(image)
    
    plt.axis('off')
    plt.title('{} image'.format(class_names[3]))
    
plt.show()
plt.figure(figsize=(12, 8))

# Prepare Image to Tensor

X_type_1 = np.array(images_type_1)
X_type_2 = np.array(images_type_2)
X_type_3 = np.array(images_type_3)
X_type_4 = np.array(images_type_4)

# Check the shape using .shape() check the images count

print (X_type_1.shape)
print (X_type_2.shape)
print (X_type_3.shape)
print (X_type_4.shape)


print (X_type_2)

X = np.concatenate((X_type_1, X_type_2), axis=0)

if len(X_type_3):
    X = np.concatenate((X, X_type_3), axis=0)

if len(X_type_4):
    X = np.concatenate((X, X_type_4), axis=0)
    
# Scaling the data to 1 - 0

X = X / 255.0

print (X.shape)


y_type_1 = [0 for item in enumerate(X_type_1)]
y_type_2 = [1 for item in enumerate(X_type_2)]
y_type_3 = [2 for item in enumerate(X_type_3)]
y_type_4 = [3 for item in enumerate(X_type_4)]

y = np.concatenate((y_type_1, y_type_2), axis=0)

if len(y_type_3):
    y = np.concatenate((y, y_type_3), axis=0)

if len(y_type_4):
    y = np.concatenate((y, y_type_4), axis=0)
    
y = to_categorical(y, num_classes=len(class_names))

print(y.shape)



# Situational - values, you may not adjust these

conv_1 = 16
conv_1_drop = 0.2
conv_2 = 32
conv_2_drop = 0.2
dense_1_n = 1024
dense_1_drop = 0.2
dense_2_n = 512
dense_2_drop = 0.2

# Values you can adjust
lr = 0.001
epochs = 10
batch_size = 10
color_channels = 3

def build_model(conv_1_drop = conv_1_drop, conv_2_drop = conv_2_drop,
                dense_1_n = dense_1_n, dense_1_drop = dense_1_drop,
                dense_2_n = dense_2_n, dense_2_drop = dense_2_drop,
                lr = lr):
    
    model = Sequential()
    
    model.add(Convolution2D(conv_1, (3, 3),
                            input_shape = (width, height, color_channels),
                            activation='relu'))
    
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Dropout(conv_1_drop))
    
    # ---
    
    model.add(Convolution2D(conv_2, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(conv_1_drop))
    
    # ---
    model.add(Flatten())
    
    # ---
    model.add(Dense(dense_1_n, activation='relu'))
    model.add(Dropout(dense_1_drop))
    
    # ---
    model.add(Dense(dense_2_n, activation='relu'))
    model.add(Dropout(dense_2_drop))
    
    # ---
    model.add(Dense(len(class_names), activation='softmax'))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(clipvalue=0.5),
                  metrics=['accuracy'])
    
    return model

# model parameter

model = build_model()

model.summary()

# Do not run yet

history = model.fit(X, y, validation_split=0.10, epochs = 10, batch_size = 5)

print(history) 

# Model evaluation
scores = model.evaluate(X, y, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model loss')
plt.ylabel('loss and accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

# Prediction

def plt_show(img):
    plt.imshow(img)
    plt.show()
    
headphone = 'img_1/10.png'
mouse = 'img_2/10.png'
harddrive = 'img_3/10.png'
glasses = 'img_4/10.png'

imgs = [headphone, mouse, harddrive, glasses]

# def predict_(img_path):

classes = None
predicted_classes = []
true_labels = []

for i in range(len(imgs)):
    type_ = tf.keras.preprocessing.image.load_img(imgs[i], target_size=(width, height))
    plt.imshow(type_)
    plt.show()
    
    type_x = np.expand_dims(type_, axis=0)
    prediction = model.predict(type_x)
    index = np.argmax(prediction)

    print(class_names[index])
    classes = class_names[index]
    predicted_classes.append(class_names[index])

    true_labels.append(class_names[i % len(class_names)])
    
cm = confusion_matrix(class_names, predicted_classes)
f = sns.heatmap(cm, xticklabels=class_names, yticklabels=predicted_classes, annot=True)

type_1 = tf.keras.preprocessing.image.load_img('img_1/10.png', target_size=(width, height))

plt.imshow(type_1)
plt.show()

type_1_x = np.expand_dims(type_1, axis=0)

predictions = model.predict(type_1_x)
index = np.argmax(predictions)

print(class_names[index])

type_2 = tf.keras.preprocessing.image.load_img('img_2/10.png', target_size=(width, height))

plt.imshow(type_2)
plt.show()

type_2_x = np.expand_dims(type_2, axis=0)
predictions = model.predict(type_2_x)

index = np.argmax(predictions)
print(class_names[index])

type_3 = tf.keras.preprocessing.image.load_img('img_3/10.png', target_size=(width, height))

plt.imshow(type_3)
plt.show()

type_3_x = np.expand_dims(type_3, axis=0)
predictions = model.predict(type_3_x)

index = np.argmax(predictions)
print(class_names[index])

type_4 = tf.keras.preprocessing.image.load_img('img_4/10.png', target_size=(width, height))

plt.imshow(type_4)
plt.show()

type_4_x = np.expand_dims(type_4, axis=0)
predictions = model.predict(type_4_x)

index = np.argmax(predictions)
print(class_names[index])

#Live Predictions using camera

CAMERA = cv2.VideoCapture(1)


while True:
    _, frame = CAMERA.read()

    # Flip
    frame = cv2.flip(frame, 1)

    # Rescale the image output
    aspect = frame.shape[1] / float(frame.shape[0])
    res = int(aspect * camera_height)  # Landscape orientation - wide image
    frame = cv2.resize(frame, (res, camera_height))

    # Get ROI
    roi = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

    # Parse BRG to RGB
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    # Adjust alignment
    roi = cv2.resize(roi, (width, height))
    roi = np.expand_dims(roi, axis=0)

    predictions = model.predict(roi)
    type_1_x, type_2_x, type_3_x, type_4_x = predictions[0]

    # Green rectangle
    cv2.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), 2)


    # Predictions/Labels
    type_1_text = '{} - {}%'.format(class_names[0], int(type_1_x * 100))
    cv2.putText(frame, type_1_text, (70, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)

    type_2_text = '{} - {}%'.format(class_names[1], int(type_2_x * 100))
    cv2.putText(frame, type_2_text, (70, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)

    type_3_text = '{} - {}%'.format(class_names[2], int(type_3_x * 100))
    cv2.putText(frame, type_3_text, (70, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)

    type_4_text = '{} - {}%'.format(class_names[3], int(type_4_x * 100))
    cv2.putText(frame, type_4_text, (70, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)

    cv2.imshow('Real-time object detection', frame)

    # Controls q = quit
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

# Release the camera
CAMERA.release()
cv2.destroyAllWindows()
