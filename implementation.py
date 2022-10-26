from imutils.video import VideoStream
import numpy as np
import cv2 as cv
import imutils
from keras.models import load_model

emotion_dict = {0: "angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6 : "Surprised"}
print("Starting Video Stream---")
cap = VideoStream(src=0).start()
while True:
    frame = cap.read()
    frame = imutils.resize(frame, width = 1000)
    bounding_box = cv.CascadeClassifier('C:/Users/HP/OneDrive/Desktop/Emojify/haarcascade_frontalcatface.xml')
    num_faces = bounding_box.detectMultiScale(frame, scaleFactor = 1.3, minNeighbors = 5)
    
    for(x, y, w, h) in num_faces:
        cv.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 2)
        roi_frame = frame[y:y+h, x: x+w]
        cropped_img = np.expand_dims(np.expand_dims(cv.resize(roi_frame, 48)), -1, 0)
        emotion_prediction = load_model("emojify.model")
        emotion_prediction = emotion_prediction(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        cv.putText(frame, emotion_dict[maxindex], (x+20), (y-60), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2), cv.LINE_AA
        
    cv.imshow("Frame", frame)
    key = cv.waitKey(1) & 0xFF
        
    if key == ord("q"):
        break
        
cv.destroyAllWindows()
cap.stop()
        



        