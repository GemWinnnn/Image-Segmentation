#Live Predictions using camera
from keras.applications import inception_v3

import time

CAMERA = cv2.VideoCapture (0)
camera_height = 500

while(True):
    _, frame = CAMERA.read()

    #Flip
    frame = cv2.flip(frame, 1)

    #Rescale the images output
    aspect = frame.shape[1] / float (frame.shape[0])
    res = int(aspect* camera_height)
    frame = cv2.resize(frame, (res, camera_height))

    #Get roi
    roi = frame[75+2:425-2, 300+2:650-2]
    
    #Parse BRG to RGB
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    #Adjust alignment
    roi = cv2.resize(roi, (width, height))
    roi_x = np.expand_dims (roi, axis=0)

    predictions = model.predict(roi_x)
    type_1_x, type_2_x, type_3_x, type_4_x = predictions[0]

    #The green rectable
    cv2.rectangle(frame, (300, 75), (650, 425), (240,100,0), 2)

    #Predictions / Labels
    tipe_1_txt = '{} - {}%'.format(class_names[0], int (type_1_x*100))
    cv2.putText(frame, tipe_1_txt, (70, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240,240,240), 2)
    
    tipe_2_txt = '{} - {}%'.format(class_names[1], int (type_2_x*100))
    cv2.putText(frame, tipe_2_txt, (70, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240,240,240), 2)
    
    tipe_3_txt = '{} - {}%'.format(class_names[2], int (type_3_x*100))
    cv2.putText(frame, tipe_3_txt, (70, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240,240,240), 2)
    
    tipe_4_txt = '{} - {}%'.format(class_names[3], int (type_4_x*100))
    cv2.putText(frame, tipe_4_txt, (70, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240,240,240), 2)
    
    cv2.imshow("Real time object detection", frame)

    #Controls q = quit/ s = capturing
    key = cv2.waitKey(1)
    if key & 0xff☐☐ ord('q');
        break

    #preview
    plt.imshow(frame) 
    plt.show()

    #Camera
    CAMERA.release()
    cv2.destroyAllWindows()