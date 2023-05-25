import cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import Adam
import seaborn as sns
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical
import tensorflow as tf


class ObjectDetectionModel:
    def __init__(self):
        self.class_names = ['HEADPHONE', 'HARDDRIVE', 'MOUSE', 'GLASSES']
        self.camera_height = 500
        self.save_width = 200
        self.save_height = 200
        self.width = 200
        self.height = 200
        self.CAMERA = None

        # Create an empty list for each type of frame
        self.raw_frames_type_1 = []
        self.raw_frames_type_2 = []
        self.raw_frames_type_3 = []
        self.raw_frames_type_4 = []

        # Initialize empty lists to store images of each type
        self.images_type_1 = []
        self.images_type_2 = []
        self.images_type_3 = []
        self.images_type_4 = []

        # Model parameters
        self.conv_1 = 16
        self.conv_1_drop = 0.2
        self.conv_2 = 32
        self.conv_2_drop = 0.2
        self.dense_1_n = 1024
        self.dense_1_drop = 0.2
        self.dense_2_n = 512
        self.dense_2_drop = 0.2
        self.lr = 0.001
        self.epochs = 15
        self.batch_size = 20
        self.color_channels = 3
        self.model = None

    def initialize_camera(self):
        self.CAMERA = cv2.VideoCapture(0)

    def capture_frames(self):
        while self.CAMERA.isOpened():
            # Read a new camera frame
            ret, frame = self.CAMERA.read()

            # Flip the frame horizontally
            frame = cv2.flip(frame, 1)

            # Rescale the image output
            aspect = frame.shape[1] / float(frame.shape[0])
            res = int(aspect * self.camera_height)
            frame = cv2.resize(frame, (res, self.camera_height))

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
            cv2.imshow('Capturing', frame)

            # Controls q = quit/ s = capturing
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('1'):
                # Save the raw frames to frame
                self.raw_frames_type_1.append(frame)
                print("Captured type 1 frame.")
            elif key == ord('2'):
                self.raw_frames_type_2.append(frame)
                print("Captured type 2 frame.")
            elif key == ord('3'):
                self.raw_frames_type_3.append(frame)
                print("Captured type 3 frame.")
            elif key == ord('4'):
                self.raw_frames_type_4.append(frame)
                print("Captured type 4 frame.")

        # Release the camera
        self.CAMERA.release()
        cv2.destroyAllWindows()

    def crop_and_save_images(self):
      
        # Calculate the coordinates using the desired formula or logic
        center_x = self.camera_height // 2
        center_y = self.camera_height // 2
        rect_width = 350
        rect_height = 350


        top_left_x = center_x - rect_width // 2
        top_left_y = center_y - rect_height // 2
        bottom_right_x = center_x + rect_width // 2
        bottom_right_y = center_y + rect_height // 2

        for i, frame in enumerate(self.raw_frames_type_1):
            # Get ROI
            roi = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

            # Parse BGR to RGB
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

            # Resize to save_width x save_height
            roi = cv2.resize(roi, (self.save_width, self.save_height))

            # Save
            cv2.imwrite('img_1/{}.png'.format(i), cv2.cvtColor(roi, cv2.COLOR_RGB2BGR))

            plt.imshow(roi)
            plt.axis('off')
            plt.show()

        for i, frame in enumerate(self.raw_frames_type_2):
            # Get ROI
            roi = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

            # Parse BGR to RGB
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

            # Resize to save_width x save_height
            roi = cv2.resize(roi, (self.save_width, self.save_height))

            # Save
            cv2.imwrite('img_2/{}.png'.format(i), cv2.cvtColor(roi, cv2.COLOR_RGB2BGR))

            plt.imshow(roi)
            plt.axis('off')
            plt.show()

        for i, frame in enumerate(self.raw_frames_type_3):
            # Get ROI
            roi = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

            # Parse BGR to RGB
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

            # Resize to save_width x save_height
            roi = cv2.resize(roi, (self.save_width, self.save_height))

            # Save
            cv2.imwrite('img_3/{}.png'.format(i), cv2.cvtColor(roi, cv2.COLOR_RGB2BGR))

            plt.imshow(roi)
            plt.axis('off')
            plt.show()

        for i, frame in enumerate(self.raw_frames_type_4):
            # Get ROI
            roi = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

            # Parse BGR to RGB
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

            # Resize to save_width x save_height
            roi = cv2.resize(roi, (self.save_width, self.save_height))

            # Save
            cv2.imwrite('img_4/{}.png'.format(i), cv2.cvtColor(roi, cv2.COLOR_RGB2BGR))

            plt.imshow(roi)
            plt.axis('off')
            plt.show()

    def load_images(self):
        for image_path in glob('img_1/*.png*'):
            image = tf.keras.preprocessing.image.load_img(image_path, target_size=(self.width, self.height))
            x = tf.keras.preprocessing.image.img_to_array(image)
            self.images_type_1.append(x)

        for image_path in glob('img_2/*.png*'):
            image = tf.keras.preprocessing.image.load_img(image_path, target_size=(self.width, self.height))
            x = tf.keras.preprocessing.image.img_to_array(image)
            self.images_type_2.append(x)

        for image_path in glob('img_3/*.png*'):
            image = tf.keras.preprocessing.image.load_img(image_path, target_size=(self.width, self.height))
            x = tf.keras.preprocessing.image.img_to_array(image)
            self.images_type_3.append(x)

        for image_path in glob('img_4/*.png*'):
            image = tf.keras.preprocessing.image.load_img(image_path, target_size=(self.width, self.height))
            x = tf.keras.preprocessing.image.img_to_array(image)
            self.images_type_4.append(x)

    def visualize_images(self):
        plt.figure(figsize=(12, 8))

        # Generate visualization for images_type_1
        for i, x in enumerate(self.images_type_1[:5]):
            plt.subplot(1, 5, i + 1)
            image = tf.keras.preprocessing.image.array_to_img(x)
            plt.imshow(image)

            plt.axis('off')
            plt.title('{} image'.format(self.class_names[0]))

        plt.show()
        plt.figure(figsize=(12, 8))

        # Generate visualization for images_type_2
        for i, x in enumerate(self.images_type_2[:5]):
            plt.subplot(1, 5, i + 1)
            image = tf.keras.preprocessing.image.array_to_img(x)
            plt.imshow(image)

            plt.axis('off')
            plt.title('{} image'.format(self.class_names[1]))

        plt.show()
        plt.figure(figsize=(12, 8))

        # Generate visualization for images_type_3
        for i, x in enumerate(self.images_type_3[:5]):
            plt.subplot(1, 5, i + 1)
            image = tf.keras.preprocessing.image.array_to_img(x)
            plt.imshow(image)

            plt.axis('off')
            plt.title('{} image'.format(self.class_names[2]))

        plt.show()
        plt.figure(figsize=(12, 8))

        # Generate visualization for images_type_4
        for i, x in enumerate(self.images_type_4[:5]):
            plt.subplot(1, 5, i + 1)
            image = tf.keras.preprocessing.image.array_to_img(x)
            plt.imshow(image)

            plt.axis('off')
            plt.title('{} image'.format(self.class_names[3]))

        plt.show()

    def prepare_data(self):
        X_type_1 = np.array(self.images_type_1)
        X_type_2 = np.array(self.images_type_2)
        X_type_3 = np.array(self.images_type_3)
        X_type_4 = np.array(self.images_type_4)

        # Check the shape using .shape() check the images count
        print(X_type_1.shape)
        print(X_type_2.shape)
        print(X_type_3.shape)
        print(X_type_4.shape)

        X = np.concatenate((X_type_1, X_type_2), axis=0)

        if len(X_type_3):
            X = np.concatenate((X, X_type_3), axis=0)

        if len(X_type_4):
            X = np.concatenate((X, X_type_4), axis=0)

        # Scaling the data to 1 - 0
        X = X / 255.0
        print(X.shape)

        y_type_1 = [0 for _ in range(len(X_type_1))]
        y_type_2 = [1 for _ in range(len(X_type_2))]
        y_type_3 = [2 for _ in range(len(X_type_3))]
        y_type_4 = [3 for _ in range(len(X_type_4))]

        y = np.concatenate((y_type_1, y_type_2), axis=0)

        if len(y_type_3):
            y = np.concatenate((y, y_type_3), axis=0)

        if len(y_type_4):
            y = np.concatenate((y, y_type_4), axis=0)

        y = to_categorical(y, num_classes=len(self.class_names))
        print(y.shape)

        return X, y

    def build_model(self):
        model = Sequential()

        model.add(Conv2D(self.conv_1, (3, 3),
                         input_shape=(self.width, self.height, self.color_channels),
                         activation='relu'))

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Dropout(self.conv_1_drop))

        model.add(Conv2D(self.conv_2, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(self.conv_2_drop))

        model.add(Flatten())

        model.add(Dense(self.dense_1_n, activation='relu'))
        model.add(Dropout(self.dense_1_drop))

        model.add(Dense(self.dense_2_n, activation='relu'))
        model.add(Dropout(self.dense_2_drop))

        model.add(Dense(len(self.class_names), activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(clipvalue=0.5),
                      metrics=['accuracy'])

        self.model = model
        self.model.summary()

    def train_model(self, X, y):
        history = self.model.fit(X, y, validation_split=0.10, epochs=self.epochs, batch_size=self.batch_size)
        return history

    def evaluate_model(self, X, y, history):
        scores = self.model.evaluate(X, y, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1] * 100))

        # Plot training and validation accuracy/loss using the provided history object
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        plt.show()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        plt.show()

        plt.plot(history.history['accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper right')
        plt.show()

    def predict_classes(self, img_paths):
        def plt_show(img):
            plt.imshow(img)
            plt.show()

        # Define the image paths
        headphone = 'img_1/10.png'
        mouse = 'img_2/10.png'
        harddrive = 'img_3/10.png'
        glasses = 'img_4/10.png'

        imgs = [headphone, mouse, harddrive, glasses]

        # Initialize lists to store predicted and true labels
        predicted_classes = []
        true_labels = []

        # Predict classes for each image
        for i in range(len(imgs)):
            img_path = imgs[i]
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(self.width, self.height))
            plt_show(img)
            
            img_x = np.expand_dims(img, axis=0)
            predictions = self.model.predict(img_x)
            index = np.argmax(predictions)

            print(self.class_names[index])
            predicted_classes.append(self.class_names[index])

            true_labels.append(self.class_names[i % len(self.class_names)])
            
        # Calculate confusion matrix
        cm = confusion_matrix(true_labels, predicted_classes)
        f = sns.heatmap(cm, xticklabels=self.class_names, yticklabels=self.class_names, annot=True)

    def evaluate_images(self):
        type_1 = tf.keras.preprocessing.image.load_img('img_1/10.png', target_size=(self.width, self.height))
        plt.imshow(type_1)
        plt.show()

        type_1_x = np.expand_dims(type_1, axis=0)
        predictions = self.model.predict(type_1_x)
        index = np.argmax(predictions)
        print(self.class_names[index])

        type_2 = tf.keras.preprocessing.image.load_img('img_2/10.png', target_size=(self.width, self.height))
        plt.imshow(type_2)
        plt.show()

        type_2_x = np.expand_dims(type_2, axis=0)
        predictions = self.model.predict(type_2_x)
        index = np.argmax(predictions)
        print(self.class_names[index])

        type_3 = tf.keras.preprocessing.image.load_img('img_3/10.png', target_size=(self.width, self.height))
        plt.imshow(type_3)
        plt.show()

        type_3_x = np.expand_dims(type_3, axis=0)
        predictions = self.model.predict(type_3_x)
        index = np.argmax(predictions)
        print(self.class_names[index])

        type_4 = tf.keras.preprocessing.image.load_img('img_4/10.png', target_size=(self.width, self.height))
        plt.imshow(type_4)
        plt.show()

        type_4_x = np.expand_dims(type_4, axis=0)
        predictions = self.model.predict(type_4_x)
        index = np.argmax(predictions)
        print(self.class_names[index])

    def live_predictions(self):
        self.initialize_camera()

        while True:
            ret, frame = self.CAMERA.read()
            if not ret:
                break

            # Flip
            frame = cv2.flip(frame, 1)

            # Rescale the image output
            aspect = frame.shape[1] / float(frame.shape[0])
            res = int(aspect * self.camera_height)  # Landscape orientation - wide image
            frame = cv2.resize(frame, (res, self.camera_height))

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

            # Get ROI
            roi = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

            # Parse BRG to RGB
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

            # Adjust alignment
            roi = cv2.resize(roi, (self.width, self.height))
            roi = np.expand_dims(roi, axis=0)

            predictions = self.model.predict(roi)
            type_1_x, type_2_x, type_3_x, type_4_x = predictions[0]

            # Green rectangle
            cv2.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), 2)

            # Predictions/Labels
            type_1_text = '{} - {}%'.format(self.class_names[0], int(type_1_x * 100))
            cv2.putText(frame, type_1_text, (70, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)

            type_2_text = '{} - {}%'.format(self.class_names[1], int(type_2_x * 100))
            cv2.putText(frame, type_2_text, (70, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)

            type_3_text = '{} - {}%'.format(self.class_names[2], int(type_3_x * 100))
            cv2.putText(frame, type_3_text, (70, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)

            type_4_text = '{} - {}%'.format(self.class_names[3], int(type_4_x * 100))
            cv2.putText(frame, type_4_text, (70, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)

            cv2.imshow('Real-time object detection', frame)

            # Controls q = quit
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break

        # Release the camera
        self.CAMERA.release()
        cv2.destroyAllWindows()

    def run(self):
        self.initialize_camera()
        self.capture_frames()
        self.crop_and_save_images()
        self.load_images()
        self.visualize_images()
        X, y = self.prepare_data()
        self.build_model()
        history = self.train_model(X, y)
        self.evaluate_model(X, y, history)
        
        # Add prediction and evaluation
        imgs = ['img_1/10.png', 'img_2/10.png', 'img_3/10.png', 'img_4/10.png']
        self.predict_classes(imgs)
        self.evaluate_images()

        self.live_predictions()

if __name__ == '__main__':
    model = ObjectDetectionModel()
    model.run()
