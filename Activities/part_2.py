import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from glob import glob
from keras import preprocessing

#Prepare Image to Tensor

X_type_1 = np.array(images_type_1)
X_type_2 = np.array(images_type_2)
X_type_3 = np.array(images_type_3)
X_type_4 = np.array(images_type_4)

#Check the shape using .shape() check the images count

print(X_type_1.shape)
print(X_type_2.shape)
print(X_type_3.shape)
print(X_type_4.shape)

X_type_2

X = np.concatenate((X_type_1, X_type_2), axis = 0)

if len (X_type_3):
    X = np.concatenate((X, X_type_3), axis = 0)

if len (X_type_4):
    X = np.concatenate((X, X_type_4), axis = 0)

#Scaling the data to 1 - 0

X = X / 255.0

X.shape

from keras.utils import to_categorical

y_type_1 = [ 0 for item in enumerate(X_type_1)]
y_type_2 = [ 0 for item in enumerate(X_type_2)]
y_type_3 = [ 0 for item in enumerate(X_type_3)]
y_type_4 = [ 0 for item in enumerate(X_type_4)]

y = np.concatenate((y_type_1, y_type_2), axis = 0)

if len(y_type_3):
    y = np.concatenate((y, y_type_3), axis = 0)

if len(y_type_4):
    y = np.concatenate((y, y_type_3), axis = 0)

y = to_categorical(y, num_classes=len(class_names))

y.shape

#CNN Config


from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Flatten, Dense
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adamax


#Default Parameters

#situational - values, you may not adjust these
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


def build_model(conv_1_drop = conv_1_drop, conv_2_drop = conv_2_drop,
                dense_1_n = dense_1_n, dense_1_drop = dense_1_drop,
                dense_2_n = dense_2_n, dense_2_drop = dense_2_drop,
                lr=lr):
    model = Sequential( )

    model.add(Convolution2D ( conv_1 , (3,3),
                             input_shape= (width, height, color_channels),
                             activation='relu'))
    

    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Dropout(conv_1_drop))

    #---

    model.add(Convolution2D( conv_2, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(conv_1_drop))

    #---
    model.add(Flatten())

    #---
    model.add(Dense(dense_1_n, activation='relu'))
    model.add(Dropout(dense_1_drop))

    #---
    model.add(Dense(dense_2_n, activation='relu'))
    model.add(Dropout(dense_2_drop))

    #---
    model.add(Dense(len(class_names), activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(clipvalue-0.5).
                  metrics=['accuracy'])
    return model


#model parameter
model = build_model()

model.summary()

#Do not run yet

history = model.fit(X,y, validation_split=0.10, epochs=10, batch_size=5)

print(history)









