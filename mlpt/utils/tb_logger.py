import os
import time
from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    def __init__(self, base_log_dir="runs/detect/tensorboard"):
        """
        Логирование общих метрик в отдельную папку.
        """
        timestamp = int(time.time())
        self.log_dir = os.path.join(base_log_dir, f"run_{timestamp}")
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        print(f"[INFO] TensorBoard logs at: {self.log_dir}")

    def log_metrics(self, model_name: str, metrics: dict, step: int = 0):
        for name, value in metrics.items():
            self.writer.add_scalar(f"{model_name}/{name}", value, step)

    def close(self):
        self.writer.close()
