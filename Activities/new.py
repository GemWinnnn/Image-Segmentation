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

class_names = ['HEADPHONE','USBDRIVE','MOUSE','POWERBANK']

# Initialize the camera
CAMERA = cv2.VideoCapture(0)

camera_height = 500
save_width = 350
save_height = 350
width = 350
height = 350

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
    
    #save
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

# Load images from 'img_1' directory
for image_path in glob('img_1/*.png'):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((width, height))
    img_array = np.array(img)
    images_type_1.append(img_array)
    # labels.append(class_index)

# Load images from 'img_2' directory
for image_path in glob('img_2/*.*'):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((width, height))
    img_array = np.array(img)
    images_type_2.append(img_array)
    # labels.append(class_index)

# Load images from 'img_3' directory
for image_path in glob('img_3/*.*'):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((width, height))
    img_array = np.array(img)
    images_type_3.append(img_array)
    # labels.append(class_index)

# Load images from 'img_4' directory
for image_path in glob('img_4/*.*'):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((width, height))
    img_array = np.array(img)
    images_type_4.append(img_array)
    # labels.append(class_index)



print('Shape of images_type_1:', images_type_1[0].shape)
print('Shape of images_type_2:', images_type_2[0].shape)
print('Shape of images_type_3:', images_type_3[0].shape)
print('Shape of images_type_4:', images_type_4[0].shape)
    
plt.figure(figsize=(12,8))

# Generate visualization for images_type_1
for i, x in enumerate(images_type_1[:5]):
    plt.subplot(1, 5, i+1)
    image = Image.fromarray(np.uint8(x))
    plt.imshow(image)
    plt.axis('off')
    plt.title('{} image'.format(class_names[0]))
plt.show()

plt.figure(figsize=(12, 8))

# Generate visualization for images_type_2
for i, x in enumerate(images_type_2[:5]):
    plt.subplot(1, 5, i+1)
    image = Image.fromarray(np.uint8(x))
    plt.imshow(image)
    plt.axis('off')
    plt.title('{} image'.format(class_names[1]))
plt.show()

plt.figure(figsize=(12, 8))

# Generate visualization for images_type_3
for i, x in enumerate(images_type_3[:5]):
    plt.subplot(1, 5, i+1)
    image = Image.fromarray(np.uint8(x))
    plt.imshow(image)
    plt.axis('off')
    plt.title('{} image'.format(class_names[2]))
plt.show()

plt.figure(figsize=(12, 8))

# Generate visualization for images_type_4
for i, x in enumerate(images_type_4[:5]):
    plt.subplot(1, 5, i+1)
    image = Image.fromarray(np.uint8(x))
    plt.imshow(image)
    plt.axis('off')
    plt.title('{} image'.format(class_names[3]))
plt.show()
plt.figure(figsize=(12,8))


# Prepare Image to Tensor
X_type_1 = np.array(images_type_1)
X_type_2 = np.array(images_type_2)
X_type_3 = np.array(images_type_3)
X_type_4 = np.array(images_type_4)


#Check the image shape using .shape()

print (X_type_1.shape)
print (X_type_2.shape)
print (X_type_3.shape)
print (X_type_4.shape)



print(X_type_2)

 
X = np.concatenate((X_type_1, X_type_2), axis=0)

if len (X_type_3):
    X = np.concatenate((X, X_type_3), axis=0)
    
if len (X_type_4):
    X = np.concatenate((X, X_type_4), axis=0)

#Scaling the data to 1-0

X = X/225.0

print (X.shape)
#(71,96,96,3)


y_type_1 = [0 for item in enumerate (X_type_1)]
y_type_2 = [1 for item in enumerate (X_type_2)]
y_type_3 = [2 for item in enumerate (X_type_3)]
y_type_4 = [3 for item in enumerate (X_type_4)]

y = np.concatenate((y_type_1,y_type_2),axis=0)

if len (y_type_3):
    y = np.concatenate((y,y_type_3),axis=0)
    
if len (y_type_4):
    y = np.concatenate((y,y_type_4),axis=0)
    
y = to_categorical(y,num_classes=len(class_names))

print(y.shape)
#(72, 4)

#default Parameters

conv_1 = 16
conv_1_drop = 0.2

conv_2 = 32
conv_2_drop = 0.2

dense_1_n = 1024
dense_1_drop = 0.2

dense_2_n =512
dense_2_drop = 0.2

lr = 0.001
epochs = 15
batch_size =10
color_channels =  3


def build_model (conv_1_drop = conv_1_drop, conv_2_drop = conv_2_drop, 
                 dense_1_drop = dense_1_drop, dense_1_n = dense_1_n,
                 dense_2_n = dense_2_n, dense_2_drop = dense_2_drop,
                 lr = lr):
    
    model = Sequential()
    
    model.add(Convolution2D(conv_1,(3,3),input_shape = (width,height, color_channels),
                            activation = 'relu'))
    
    model.add (MaxPooling2D(pool_size=(2,2)))
    
    model.add (Dropout(conv_1_drop))
    

    model.add(Convolution2D(conv_2,(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add (Dropout(conv_2_drop))
    
    
    model.add(Flatten())
    
    model.add(Dense(dense_1_n , activation='relu'))
    model.add(Dropout(dense_1_drop))
    
    
    model.add(Dense(dense_2_n , activation='relu'))
    model.add(Dropout(dense_2_drop))
    
    model.add (Dense(len(class_names),activation='softmax'))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer = Adam(clipvalue=0.5),
                  metrics=['accuracy'])
    
    return model

model = build_model()

model.summary()

history = model.fit(X,y, validation_split=0.10,epochs=epochs,batch_size=5)

print(history)

#model evaluation

scores = model.evaluate(X, y, verbose='verbose')
print("Accuracy: %.2f%%" % (scores[1]*100))

# Accuracy: 100.00%

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

#Prediction
def plt_show(img):
    plt.imshow(img)
    plt.show()

headphone = 'img_1/0.png'
mouse = 'img_2/0.png'
usbdrive = 'img_3/0.png'
powerbank = 'img_4/0.png'

imgs = [headphone, mouse, usbdrive, powerbank]

#def predict_(img_path):
classes = None
predicted_classes = []

for i in range(len( imgs)):
    type_ = tf.keras.utils.load_img(imgs[i], target_size=(width, height))
    plt.imshow(type_) # type: ignore
    plt.show()

    type_x = np.expand_dims(type_, axis=0) # type: ignore
    prediction = model.predict(type_x)
    index = np.argmax(prediction)
    
    print(class_names[index])
    classes = class_names[index]
    predicted_classes.append(class_names[index])

cm = confusion_matrix(class_names, predicted_classes)
f = sns.heatmap(cm, xticklabels=" ".join(class_names), yticklabels=" ".join(predicted_classes), annot=True)




#Live Predictions using camera

CAMERA = cv2.VideoCapture(0)
camera_height = 500

while(True):
  
    res, frame = CAMERA.read()

    frame = cv2.flip (frame, 1)

    #Resacle the images output
    aspect = frame.shape[1]/float(frame.shape[0])
    res = int(aspect* camera_height)
    frame = cv2.resize(frame, (res, camera_height))

    x1 = int(frame.shape[1] * 0.25)
    y1 = int(frame.shape[0] * 0.25)
    
    x2 = int(frame.shape[1] * 0.75)
    y2 = int(frame.shape[0] * 0.75)

    roi = frame[y1+2:y2-2, x1+2:x2-2]
    #roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi = cv2.resize(roi, (width, height))
    roi_x = np.expand_dims(roi, axis=0)

    predictions = model.predict(roi_x)
    type_1_x, type_2_x, type_3_x, type_4_x = predictions[0]

    #The green rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), (240, 100, 0), 2)

    #Predictions / Labels
    type_1_txt = '{} - {}%'.format(class_names[0], int(type_1_x*100))
    cv2.putText(frame, type_1_txt, (70, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(240,240,240), 2)

    type_2_txt = '{} - {}%'.format(class_names[1], int(type_2_x*100))
    cv2.putText(frame, type_2_txt, (70, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(240,240,240), 2)

    type_3_txt = '{} - {}%'.format(class_names[2], int(type_3_x*100))
    cv2.putText(frame, type_3_txt, (70, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(240,240,240), 2)

    type_4_txt = '{} - {}%'.format(class_names[3], int(type_4_x*100))
    cv2.putText(frame, type_4_txt, (70, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(240,240,240), 2)

    cv2.imshow("Real time object detection", frame)
    key = cv2.waitKey(1)
    if key & 0xff == ord('q'):
        break

    #preview
    plt.imshow(frame)
    plt.show()

    #Camera
CAMERA.release()
cv2.destroyAllWindows()