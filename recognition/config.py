from dataclasses import dataclass

@dataclass
class jet:
    dataset: str = "gtsrb-german-traffic-sign"
    model: str = "trafficsignnet.model"
    # Гиперпараметры
    batch_size: int = 64
    learning_rate: float = 0.005 # Скорость обучения η
    epoch_nums: int = 30

    @staticmethod # для интерактивной консоли
    def reload():
        import sys
        if "config" in  sys.modules:
            del(sys.modules["config"])
#        from config import jet