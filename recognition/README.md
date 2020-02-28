## Где лежат большие файлы
### [датасет](https://yadi.sk/d/_vErSJxl6654bg) :
содержимое zip архивов:
- `image.zip` - фото из оригинального датасета
- `csv.zip` - файлы описаний оттуда же
- `npy.zip` - обработанные фото для тренировки модели

распаковать в папку проекта `gtsrb-german-traffic-sign` (jet.dataset)

для запуска 2_train.py, достаточно `npy.zip`

## Дополнительные программы:
### Стробоскоп 
файл `3_testcam.py` за основу + 
- сохранение:
```python
p = os.path.sep.join([os.getcwd(), "images", "{}.png".format(i)])
cv2.imwrite(p, image)
```
 - задержка:
 ```python
 import time
 time.sleep(секунды)
 ```
## Файлы:
---
## 1_preprocess.py
Подготовка датасета:
- Считываем фото с диска
- Обрабатываем

Файл настроен для демонстрации. *Перед работой* снять комментарии с 23-24 строки (`np.save(f"_{destination}X", np.array(data))` и следующей)

## 2_train.py

## preprocess.pу

## config.pу

---
## TODO:
### Загрузить одну фотку с помощью cv & skimage
- отличаются ли они?
- есть проблемы с RGB & BGR?
    - может ли захватывать с камеры бибиотекой skimage?
    - импортирвать с диска  с помощью cv?
## Заметки:
### Интерактивная консоль дополняет путь директорией, из которой jupyter notebook запущен
- Запускаем JPnotebook из recognition папки проекта
- или:
    ``` python
    import sys
    sys.path.append("C:\\Users\\eugen\\Documents\\GitHub\\jetcar\\recognition")
    ```
### Настройки keras:
    `user/.keras/keras.json`

### Функция `skimage.transform.resize`
меняет min & max т.е. нормализация не нужна

что за странный диапазон она отдает?

## Depricated
- `H = model.fit_generator(`

WARNING:tensorflow:From <ipython-input-140-b12f63d35c7b>:7: Model.fit_generator (from tensorflow.python.keras.engine.training) is deprecated and will be removed in a future version.
Instructions for updating:
Please use Model.fit, which supports generators.

WARNING:tensorflow:sample_weight modes were coerced from ... to ['...']

- `model.save(jet.model)`

WARNING:tensorflow:From C:\Users\eugen\Anaconda3\envs\tfgpu\lib\site-packages\tensorflow_core\python\ops\resource_variable_ops.py:1786: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
Instructions for updating:
If using Keras pass *_constraint arguments to layers.
INFO:tensorflow:Assets written to: trafficsignet.model1\assets



