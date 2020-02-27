from skimage import io
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from config import jet
from preprocess import preprocess

def convert_set(csvFileName, destination):
    """Функция читает картинки датасета, преобразует их и сохраняет на диск, как ".npy"""
    data = []
    
    csv = pd.read_csv(csvFileName)
    # Перемешиваем фрейм и возвращаем его весь (frac=1)
    csv = csv.sample(frac=0.01).reset_index(drop=True)
    # Обёртка tqdm - прогрессбар
    for imagePath in tqdm(csv.Path):
        # Читаем изображение
        image = io.imread(imagePath)
        # Добавляем к массиву предобработанное изображение
        data.append(preprocess(image))
    
    #np.save(f"_{destination}X", np.array(data))
    #np.save(f"_{destination}Y", csv.ClassId)

# устанавливаем папку с датасетом, корневой директорией
os.chdir(os.path.sep.join([os.getcwd(), jet.dataset]))

# вызываем функцию конвертации для тренировочного и тестового набора
print(convert_set("Train.csv", "train"))
print(convert_set("Test.csv", "test"))