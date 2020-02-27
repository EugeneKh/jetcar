from dataclasses import dataclass

@dataclass
class jet:
    dataset: str = "gtsrb-german-traffic-sign"
    model: str = "trafficsignet.model"
    # Гиперпараметры
    batch_size: int = 64
    learning_rate: float = 0.001 # Скорость обучения η
    epoch_nums: int = 10

    @staticmethod # для интерактивной консоли
    def reload():
        import sys
        if "config" in  sys.modules:
            del(sys.modules["config"])
#        from config import jet