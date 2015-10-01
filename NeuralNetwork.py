import numpy as np
import random
import math


class NeuralNetwork:
    def __init__(self, images, p, n):
        self.images = images
        self.p = p
        self.n = n
        self.weights = np.zeros((self.n, self.p), dtype=np.float64)
        self.weights_ = np.zeros((self.p, self.n), dtype=np.float64)
        self.fill_weights()

    def fill_weights(self):
        for i in range(0, self.n):
            for j in range(0, self.p):
                self.weights[i, j] = self.weights_[j, i] = random.random()

        self.normalize_weights()

    def direct_conversion(self, x):
        y = np.zeros(self.p, dtype=np.float64)

        for j in range(0, self.p):
            for i in range(0, self.n):
                y[j] += self.weights[i, j] * x[i]

        return y

    def inverse_conversion(self, y):
        x = np.zeros(self.n, dtype=np.float64)

        for i in range(0, self.n):
            for j in range(0, self.p):
                x[i] += self.weights_[j, i] * y[j]

        return x

    def get_error(self, x, x_):
        e = 0

        for i in range(0, self.n):
            e += pow(x_[i] - x[i], 2)

        return e / 2

    def normalize_direct_weights(self):
        for i in range(0, self.n):
            s = 0
            for j in range(0, self.p):
                s += self.weights[i, j]**2

            s = math.sqrt(s)

            for j in range(0, self.p):
                self.weights[i, j] /= s

    def normalize_inverse_weights(self):
        for j in range(0, self.p):
            s = 0
            for i in range(0, self.n):
                s += self.weights_[j, i]**2

            s = math.sqrt(s)

            for i in range(0, self.n):
                self.weights_[j, i] /= s

    def normalize_weights(self):
        self.normalize_direct_weights()
        self.normalize_inverse_weights()

    def compute_weights(self, ii, oi, ii_, gamma, alpha, alpha_):
        for i in range(0, self.n):
            for j in range(0, self.p):
                self.weights[i, j] -= alpha * gamma[j] * ii[i]
                self.weights_[j, i] -= alpha_ * oi[j] * (ii_[i] - ii[i])

        self.normalize_weights()

    def compute_adaptive_step(self, x, size):
        s = 0

        for i in range(0, size):
            s += x[i]**2

        return 1. / s

    def train(self, inputs):
        e = 10
        iteration = 0
        gamma = np.zeros(self.p, dtype=np.float64)

        while e >= 0.01:
            e = 0
            iteration += 1

            for l in range(0, self.p):
                ii = inputs[l]
                oi = self.direct_conversion(ii)
                ii_ = self.inverse_conversion(oi)

                alpha = self.compute_adaptive_step(ii, self.n)
                alpha_ = self.compute_adaptive_step(oi, self.p)

                for j in range(0, self.p):
                    gamma[j] = 0

                    for i in range(0, self.n):
                        gamma[j] += (ii_[i] - ii[i]) * self.weights_[j, i]

                self.compute_weights(ii, oi, ii_, gamma, alpha, alpha_)

            for l in range(0, self.p):
                ii = inputs[l]
                oi = self.direct_conversion(ii)
                ii_ = self.inverse_conversion(oi)

                e += self.get_error(ii, ii_)

            print iteration, 'iteration, error =', e

    def process(self):
        self.train(self.images)

        exit(0)
