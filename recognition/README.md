## TODO:
### Загрузить одну фотку с помощью cv & skimage
- отличаются ли они?
- есть проблемы с RGB & BGR?
- может ли захватывать с камеры skimage?
- импортирвать с диска cv?
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

## Drpricated
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



---
## 1_preprocess.py
Подготовка датасета:
- Считываем фото с диска
- Обрабатываем

Файл настроен для демонстрации

Перед работой 
- снять комментарии с 23-24 стр `np.save(f"_{destination}X", np.array(data))` и следующей

---
## 2_train.py

