import numpy as np


class Window:
    def __init__(self, data, p, width, height):
        self.data = data
        self.p = p
        self.n = width * height
        self.width = width
        self.height = height
        self.weights = np.zeros((self.p, self.n), dtype=np.float64)
        self.weights_ = np.transpose(self.weights)

    def normalize_components(self, x):
        red_max = np.max(x[0, ])
        green_max = np.max(x[1, ])
        blue_max = np.max(x[2, ])

        normalized_x = np.empty((3, self.n), dtype=np.float64)

        for i in range(0, self.n):
            normalized_x[0:, i] = (2 * x[0, i] / red_max - 1), \
                                  (2 * x[0, i] / green_max - 1), \
                                  (2 * x[2, i] / blue_max - 1)

        return normalized_x

    def get_error(self, x, x_):
        return np.sum(x - x_) / 2.

    def compute_weights(self, x, y):
        for i in range(0, self.p):
            for j in range(0, self.n):
                self.weights[i, j] = self.weights[i, j] + x[j] * y[i]

    def compute_component(self, l, x):
        _in = x[l, ]

        while True:
            _out = np.dot(self.weights, _in)
            self.compute_weights(_in, _out)
            in_ = np.dot(self.weights_, _out)

            e = self.get_error(_in, in_)
            if e <= 0.01:
                break

    def process(self):
        x = np.empty((3, self.n), dtype=np.float64)

        for j in range(0, self.height):
            for i in range(0, self.width):
                x[0:, j * self.width + i] = self.data[i, j]

        x = self.normalize_components(x)

        for l in range(0, 3):
            self.compute_component(l, x)

        exit(0)
