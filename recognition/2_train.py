import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from jetcnn_v1.trafficsignnet import TrafficSignNet
import matplotlib
matplotlib.use("Agg")
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from config import jet
import pandas as pd

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
numLabels = len(np.unique(trainY))
trainY = to_categorical(trainY, numLabels)
testY = to_categorical(testY, numLabels)

# получаем множители для каждого класса, чтобы выровнять dataset
# массив с кол-вом картинок в каждом классе
classTotals = trainY.sum(axis=0)
# массив кооф ...
classWeight = classTotals.max() / classTotals
classWeight = dict(enumerate(classWeight))

# настраиваем генератор для аугментации
aug = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    horizontal_flip=False,
    vertical_flip=False,
    fill_mode="nearest")

# настраиванм оптимизатор, строим и компилим модель
opt = Adam(lr=jet.learning_rate, decay=jet.learning_rate / (jet.epoch_nums * 0.5))
model = TrafficSignNet.build(width=32, height=32, depth=3, classes=numLabels)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Обучение
H = model.fit_generator(
    aug.flow(trainX, trainY, batch_size=jet.batch_size),
    validation_data=(testX, testY),
    steps_per_epoch=trainX.shape[0] // jet.batch_size,
    epochs=jet.epoch_nums,
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
