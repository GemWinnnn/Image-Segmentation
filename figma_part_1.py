import matplotlib.pyplot as plt
import numpy as np 
import cv2

class_names = ['CUP','SPOON', 'FORK', 'MOUSE']

#Creating Realtime Dataset

CAMERA = cv2.VideoCapture(0)
camera_height = 500 

raw_frame_type_1 = []
raw_frame_type_2 = []
raw_frame_type_3 = []
raw_frame_type_4 = []

while CAMERA.isOpened(): 
    # Read a new camera frame 
    ret, frame = CAMERA.read()

    # Flip 
    frame = cv2.flip(frame,1)

    # Rescale the images output
    aspect = frame.shape[1] / float(frame.shape[0])
    res = int(aspect* camera_height)
    frame = cv2.resize(frame,(res,camera_height))

    # The green rectangle 
    cv2.rectangle(frame, (300, 75), (650, 425), (0, 255, 0), 2)

    # Show the frame 
    cv2.imshow("Capturing", frame)

    # Controls q = quit / s = capturing 
    key = cv2.waitKey(1) & 0xFF 

    if key & 0xFF == ord('q'):
        break 

    elif key & 0xFF == ord('1'): 
        # Save the raw frames to frame 
        raw_frame_type_1.append(frame)

        # Preview 
        plt.imshow(frame) 
        plt.show()

    elif key & 0xFF == ord('2'): 
        # Save the raw frames to frame 1111223344444444
        raw_frame_type_2.append(frame)

        # Preview 
        plt.imshow(frame) 
        plt.show()

    elif key & 0xFF == ord('3'): 
        # Save the raw frames to frame 
        raw_frame_type_3.append(frame)

        # Preview 
        plt.imshow(frame) 
        plt.show()

    elif key & 0xFF == ord('4'):
        # Save the raw frames to frame 
        raw_frame_type_4.append(frame)

        # Preview 
        plt.imshow(frame) 
        plt.show()

#Camera 
CAMERA.realease() 
cv2.destroyAllWindows()

save_width =399 
save_height = 400 

import os 
from glob import glob

retval = os.getcwd () 
print('Current working directory %s' % retval)

print('img1:',len(raw_frame_type_1))
print('img2:',len(raw_frame_type_2))
print('img3:',len(raw_frame_type_3))
print('img4:',len(raw_frame_type_4))

#Crop the images 
for i, frame in enumerate (raw_frame_type_1): 
    #get roi 
    roi = frame [ 75 + 2:425-2, 300 + 2:650-2]

    #Parse BRG to RGB 
    roi = cv2.cvtColor(roi,cv2.COLOR_BGR2RGB)

    #Resize to 224 x 224 
    roi = cv2.resize (roi, (save_width,save_height))

    #save
    cv2.imwrite('img_1/{}.png'.format(i), cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)) 

for i, frame in enumerate (raw_frame_type_2): 
    
    #get roi 
    roi = frame [ 75 + 2:425-2, 300 + 2:650-2]

    #Parse BRG to RGB 
    roi = cv2.cvtColor(roi,cv2.COLOR_BGR2RGB)

    #Resize to 224 x 224 
    roi = cv2.resize (roi, (save_width,save_height))

    #save
    cv2.imwrite('img_2/{}.png'.format(i), cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)) 

for i, frame in enumerate (raw_frame_type_3): 
    
    #get roi 
    roi = frame [ 75 + 2:425-2, 300 + 2:650-2]

    #Parse BRG to RGB 
    roi = cv2.cvtColor(roi,cv2.COLOR_BGR2RGB)

    #Resize to 224 x 224 
    roi = cv2.resize (roi, (save_width,save_height))

    #save
    cv2.imwrite('img_3/{}.png'.format(i), cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)) 

for i, frame in enumerate (raw_frame_type_4): 
    
    #get roi 
    roi = frame [ 75 + 2:425-2, 300 + 2:650-2]

    #Parse BRG to RGB 
    roi = cv2.cvtColor(roi,cv2.COLOR_BGR2RGB)

    #Resize to 224 x 224 
    roi = cv2.resize (roi, (save_width,save_height))

    #save
    cv2.imwrite('img_4/{}.png'.format(i), cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)) 

from glob import glob 
from keras import preprocessing
 
width = 96
height = 96 

images_type_1 = []
images_type_2 = []
images_type_3 = []
images_type_4 = []

for image_path in glob ('img_1*.*'): 
    image = preprocessing.image.load_img(image_path, target_size =(width,height))
    x = preprocessing.image.img_to_array(image) 

    images_type_1.append(x)

for image_path in glob ('img_2*.*'): 
    image = preprocessing.image.load_img(image_path, target_size =(width,height))
    x = preprocessing.image.img_to_array(image) 

    images_type_2.append(x)

for image_path in glob ('img_3*.*'): 
    image = preprocessing.image.load_img(image_path, target_size =(width,height))
    x = preprocessing.image.img_to_array(image) 

    images_type_3.append(x)

for image_path in glob ('img_4*.*'): 
    image = preprocessing.image.load_img(image_path, target_size =(width,height))
    x = preprocessing.image.img_to_array(image) 

    images_type_4.append(x)

plt.figure(figsize = (12,8)) 
 
for i, x in enumerate (images_type_1[:5]): 
    plt.subplot (1,5, i + 1)
    image = preprocessing.image.array_to_img(x) 
    plt.imshow(image)

    plt.axis ('off')
    plt.title ('{} image'.format(class_names[0]))

plt.show()

for i, x in enumerate (images_type_1[:5]): 
    plt.subplot (1,5, i + 1)
    image = preprocessing.image.array_to_img(x) 
    plt.imshow(image)

    plt.axis ('off')
    plt.title ('{} image'.format(class_names[1]))

plt.show()

for i, x in enumerate (images_type_1[:5]): 
    plt.subplot (1,5, i + 1)
    image = preprocessing.image.array_to_img(x) 
    plt.imshow(image)

    plt.axis ('off')
    plt.title ('{} image'.format(class_names[2]))

plt.show()

for i, x in enumerate (images_type_1[:5]): 
    plt.subplot (1,5, i + 1)
    image = preprocessing.image.array_to_img(x) 
    plt.imshow(image)

    plt.axis ('off')
    plt.title ('{} image'.format(class_names[3]))

plt.show()

