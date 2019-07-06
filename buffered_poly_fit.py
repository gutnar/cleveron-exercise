import numpy as np


class BufferedPolyFit:
    def __init__(self, order, size=5, max_skip=3):
        self.order = order
        self.size = size
        self.max_skip = max_skip
        self.x_buffer = []
        self.y_buffer = []
        self.skipped = 0
        self.coeffs = None
    
    def get_buffered_fit(self):
        if not len(self.x_buffer):
            return None
        
        self.coeffs = np.polyfit(
            np.concatenate(self.x_buffer),
            np.concatenate(self.y_buffer),
            self.order,
            #w=sum([[i+1]*len(self.x_buffer[i]) for i in range(len(self.x_buffer))], [])
        )
        
        return np.poly1d(self.coeffs)

    def fit(self, x, y):
        current_fit = self.get_buffered_fit()

        if not len(x):
            return current_fit
        
        self.x_buffer.append(x)
        self.y_buffer.append(y)

        if len(self.x_buffer) > self.size:
            self.x_buffer.pop(0)
            self.y_buffer.pop(0)

        return self.get_buffered_fit()
    
    def get_curvature(self, x):
        if self.coeffs == None:
            return 0
        
        d = 0
        dd = 0

        for i in range(1, self.order):
            pass
        
        return (1 + self.coeffs[2])
