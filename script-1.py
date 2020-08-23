# Importing Libraries
import time
import cv2
import pickle
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model('data/inceptionV3-model.h5')

# Necessary values
with open('data/category2label.pkl', 'rb') as pf:
    category2label = pickle.load(pf)

img_size = (100, 100)
colors = {0: (0, 0, 255), 1: (0, 255, 0), 2: (0, 255, 255)}

# Importing cascade classifier for face-detection
face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# Video from webcam
cap = cv2.VideoCapture(0)

start_time = time.time()
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()  
    
    if not ret:
        break
    
    frame_count += 1     # for fps
    frame = cv2.flip(frame, 1)       # Mirror the image
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    for (x, y, w, h) in faces:
        # Predict
        roi =  frame[y : y+h, x : x+w]
        data = cv2.resize(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB), img_size)
        data = data / 255.
        data = data.reshape((1,) + data.shape)
        scores = model.predict(data)
        target = np.argmax(scores, axis=1)[0]
        # Draw bounding boxes
        cv2.rectangle(img=frame, pt1=(x, y), pt2=(x+w, y+h), color=colors[target], thickness=2)
        text = "{}: {:.2f}".format(category2label[target], scores[0][target])
        cv2.putText(frame, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time
    cv2.putText(img=frame, text='FPS : ' + str(round(fps, 2)), org=(10, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,255,255), thickness=1)

    # Show the frame
    cv2.imshow('Face Mask Detection', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()