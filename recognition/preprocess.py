from skimage import transform
from skimage import exposure

def preprocess(image):
    """Ресайзит и повышает контрастность изображения
    
    Эта функция используется для подготовки изображений
    - перед тренировкой модели
    - перед распознаванием
    """

    image = transform.resize(image, (32, 32))
    # Контраст Limited Adaptive Histogram Equalization (CLAHE)
    # Нормализовано в диапазоне [0...1]
    image = exposure.equalize_adapthist(image, clip_limit=0.1)
    return image
