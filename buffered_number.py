import numpy as np


class BufferedNumber:
    def __init__(self, size=5):
        self.values = []
        self.size = size
    
    def get(self, value):
        self.values.append(value)

        if len(self.values) > self.size:
            self.values.pop(0)
        
        return np.mean(self.values)
