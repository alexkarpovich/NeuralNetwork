import numpy as np


class Window:
    def __init__(self, data, p, width, height):
        self.data = data
        self.p = p
        self.n = width * height
        self.width = width
        self.height = height
        self.weights = np.empty((self.n, self.p), dtype=np.float64)
        self.weights.fill(1./(self.n * self.p))
        self.weights_ = np.transpose(self.weights)

    def compute_adaptive_step(self, x):
        return 1./np.sum(x ** 2)

    def compute_direct_weights(self, x, y, _x, _y, __x):
        for i in range(0, self.n):
            for j in range(0, self.p):
                self.weights[i, j] = self.weights[i, j] - self.compute_adaptive_step(x[0:, 0]) * _x[i, 0]

    def process(self):
        x = np.empty((3, self.n), dtype=np.float64)

        for j in range(0, self.height):
            for i in range(0, self.width):
                x[0:,j * self.width + i] = self.data[i, j]

        y = np.dot(x, self.weights)
        _x = np.dot(y, self.weights_)
        _y = np.dot(_x, self.weights)
        __x = np.dot(_y, self.weights_)

        self.compute_direct_weights(x, y, _x, _y, __x)

        exit(0)
