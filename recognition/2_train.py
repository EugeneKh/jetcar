import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
print(f"TF ver. = {tf.version.VERSION}")
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
#   raise SystemError("GPU device not found")
    print("GPU device not found")
print(f"Found GPU at: {device_name}")

#from tensorflow.keras.utils import to_categorical
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from sklearn.metrics import classification_report
#os.chdir("c:\\Users\\eugen\\Documents\\GitHub\\jetcar\\recognition")
from jetcnn_v1.trafficsignnet import TrafficSignNet
from config import jet



# Загружаем датасет
start_cwd = os.getcwd()
os.chdir(os.path.sep.join([os.getcwd(), jet.dataset]))
os.getcwd()
trainX = np.load("trainX.npy")
trainY = np.load("trainY.npy")
testX = np.load("testX.npy")
testY = np.load("testY.npy")
os.chdir(start_cwd)
os.getcwd()

# one-hot encode
# приведение к виду [0, 0, 1, 0, 0] для 2-й из 5 категорий
# numLabels - лишний. keras сам считает. но это не точно )
numLabels = len(np.unique(trainY))
trainY = keras.utils.to_categorical(trainY, numLabels)
testY = keras.utils.to_categorical(testY, numLabels)

# получаем множители для каждого класса, чтобы выровнять dataset 
# массив с кол-вом картинок в каждом классе
classTotals = trainY.sum(axis=0)
# массив кооф ...
classWeight = classTotals.max() / classTotals
classWeight = dict(enumerate(classWeight))

# настраиваем генератор для аугментации
aug = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    horizontal_flip=False,
    vertical_flip=False,
    fill_mode="nearest")

# настраиванм оптимизатор
opt = keras.optimizers.Adam(lr=jet.learning_rate, decay=jet.learning_rate / (jet.epoch_nums * 0.5))
# строим и компилим модель
model = TrafficSignNet.build(width=32, height=32, depth=3, classes=numLabels)
#model.get_config()
#model.summary()
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Обучение
H = model.fit(
    aug.flow(trainX, trainY, batch_size=jet.batch_size),
    validation_data=(testX, testY),
    steps_per_epoch=trainX.shape[0] // jet.batch_size,
    epochs=20,
    class_weight=classWeight)

# Оценка и вывод результатов
predictions = model.predict(testX, batch_size=jet.batch_size)
labelNames = pd.read_csv("signnames.csv").signName # список с названиями знаков
print(classification_report(testY.argmax(axis=1),
    predictions.argmax(axis=1), target_names=labelNames))

# сохраняем обученную сеть на диск
model.save(jet.model)

# TODO: Сохранить историю H
# Рисуем график с отчётом
N = np.arange(0, jet.epoch_nums)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(f"{jet.model}.png")
