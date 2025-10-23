import psutil
import torch

class SystemConfig:
    def __init__(self):
        self.system_type = self.detect_system()

        if self.system_type == "strong":
            self.PROCESS_FPS = 10
            self.IMGSZ = 640
            self.MAX_WORKERS = 4
        else:
            self.PROCESS_FPS = 5
            self.IMGSZ = 480
            self.MAX_WORKERS = 1

        print(f"System detected: {self.system_type.upper()}  "
              f"(FPS={self.PROCESS_FPS}, IMG={self.IMGSZ}, WORKERS={self.MAX_WORKERS})")

    @staticmethod
    def detect_system():
        cpu_cores = psutil.cpu_count(logical=False)
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        has_gpu = torch.cuda.is_available()
        if has_gpu or (cpu_cores >= 8 and ram_gb >= 16):
            return "strong"
        return "weak"
