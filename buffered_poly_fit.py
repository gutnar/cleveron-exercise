import numpy as np


class BufferedPolyFit:
    def __init__(self, order, size=5, max_skip=3):
        self.order = order
        self.size = size
        self.max_skip = max_skip
        self.x_buffer = []
        self.y_buffer = []
        self.skipped = 0
        self.last_fit = None
    
    def fit(self, x, y):
        if not len(x):
            return self.last_fit
        
        self.x_buffer.append(x)
        self.y_buffer.append(y)

        if len(self.x_buffer) > self.size:
            self.x_buffer.pop(0)
            self.y_buffer.pop(0)
        
        self.last_fit = np.poly1d(np.polyfit(
            np.concatenate(self.x_buffer),
            np.concatenate(self.y_buffer),
            self.order
            #w=sum([[i+1]*len(self.x_buffer[i]) for i in range(len(self.x_buffer))], [])
        ))

        return self.last_fit
    
    def get_curvature(self, x):
        if self.last_fit == None:
            return 0
        
        d = np.polyder(self.last_fit)
        dd = np.polyder(d)

        return (1 + d(x)**2)**(3/2) / dd(x)
