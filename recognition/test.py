from tensorflow.keras.models import load_model
from skimage import transform
from skimage import exposure
from skimage import io
from imutils import paths
import numpy as np
import argparse
import imutils
import random
import cv2
import os
from config import jet

# ! Убрать лишние аргументы
args = {"images":"gtsrb-german-traffic-sign/Test", "model":"trafficsignnet.model", "examples":"examples"}

# грузим модель
model = load_model(args["model"])

# load the label names
labelNames = open("signnames.csv").read().strip().split("\n")[1:]
labelNames = [l.split(",")[1] for l in labelNames]

cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,  480))
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # ! cv2 frame -> scimage image
    image = transform.resize(frame, (32, 32))
    image = exposure.equalize_adapthist(image, clip_limit=0.1)

    # preprocess the image by scaling it to the range [0, 1]
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    
    # make predictions using the traffic sign recognizer CNN
    preds = model.predict(image)
    j = preds.argmax(axis=1)[0]
    label = labelNames[j]

    cv2.putText(frame, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    out.write(frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()


    # * frame = cv.flip(frame, 0)
