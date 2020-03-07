from tensorflow.keras.models import load_model
import cv2
#import os
from config import jet
import pandas as pd
from preprocess import preprocess
import numpy as np
from skimage import transform

# Проблема с памятью (?) решение ниже
# https://github.com/tensorflow/tensorflow/issues/28326#issuecomment-565413185
import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
set_session(sess)

# грузим модель
#model = load_model(jet.model)
model = load_model("model")
model.summary()
#labelNames = pd.read_csv("signnames.csv").signName

cap = cv2.VideoCapture(0)
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,  480))
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Захватываем фрейм
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # 
    #image = preprocess(frame)
    image = transform.resize(frame, (300, 300))
    image = np.expand_dims(image, axis=0)
    
    # распознаём и присваиваем ярлык
    preds = model.predict(image)
    #print(preds[0])
    if preds[0]>0.5:
        #print("is a stop")
        label = "STOP"
    else:
        #print("is a to_right")
        label = "to_right"
    # j = preds.argmax(axis=1)[0]
    # label = labelNames[j]

    # Накладываем название на кадр
    cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    #out.write(frame)
    # выводим на экран
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
#out.release()
cv2.destroyAllWindows()