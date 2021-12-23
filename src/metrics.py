import numpy as np

from torch.nn import functional as F

class AccMeter():
    def __init__(self, num_heads=1):
        self.avg = 0
        self.n = 0
        self.num_heads = num_heads
    
    def update(self, y_pred, y_true):
        y_true = y_true.cpu().numpy().astype(int)
        y_pred = np.argmax(y_pred.cpu().numpy(), axis=1)
        last_n = self.n
        self.n += len(y_true)
        true_count = np.sum(y_true == y_pred, axis=0)
        # Incremental update
        self.avg = true_count / self.n + last_n / self.n * self.avg

    def get(self):
        return self.avg

class LossMeter():
    def __init__(self):
        self.avg = 0
        self.n = 0
    
    def update(self, value):
        self.n += 1
        self.avg = (value + self.avg * (self.n - 1)) / self.n

    def get(self):
        return self.avg