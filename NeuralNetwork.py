import numpy as np
import random
import math
import datetime


class NeuralNetwork:
    def __init__(self, images, p, n):
        """
        :param images: array of input images
        :param p: int count of output images
        :param n: int count of values in a image
        """
        self.images = images
        self.p = p
        self.n = n
        self.weights = np.zeros((self.n, self.p), dtype=np.float64)
        self.weights_ = np.zeros((self.p, self.n), dtype=np.float64)
        self.fill_weights()

    def fill_weights(self):
        """ This method fills randomly matrices of weights with value [-1, 1] """
        for i in range(0, self.n):
            for j in range(0, self.p):
                self.weights[i, j] = self.weights_[j, i] = random.uniform(-1, 1)

    def get_error(self, x, x_):
        """ This method calculate error between recovered and source images
        :param x: array - source image
        :param x_: array - recovered image
        :return: float - error between values of recovered and source images
        """
        e = 0

        for i in range(0, self.n):
            e += pow(x_[i] - x[i], 2)

        return e / 2

    def normalize_direct_weights(self):
        """ This method normalize weights of first layer """
        for i in range(0, self.n):
            s = 0
            for j in range(0, self.p):
                s += self.weights[i, j]**2

            s = math.sqrt(s)

            for j in range(0, self.p):
                self.weights[i, j] /= s

    def normalize_inverse_weights(self):
        """ This method normalize weights of second layer """
        for j in range(0, self.p):
            s = 0
            for i in range(0, self.n):
                s += self.weights_[j, i]**2

            s = math.sqrt(s)

            for i in range(0, self.n):
                self.weights_[j, i] /= s

    def normalize_weights(self):
        """ This method normalize weights of both layers """
        self.normalize_direct_weights()
        self.normalize_inverse_weights()

    def compute_weights(self, ii, oi, ii_, gamma, alpha, alpha_):
        """ This method corrects weights for first and second layers
        :param ii: array input image
        :param oi: array output image
        :param ii_: array recovered image
        :param gamma: array of gamma values
        :param alpha: float alpha for fist layer
        :param alpha_: float alpha for second layer
        """
        for i in range(0, self.n):
            for j in range(0, self.p):
                self.weights[i, j] -= alpha * gamma[j] * ii[i]
                self.weights_[j, i] -= alpha_ * oi[j] * (ii_[i] - ii[i])

        self.normalize_weights()

    def compute_adaptive_step(self, x, size):
        """ Compute adaptive step for passed layer
        :param x: array image of first of second layer
        :param size: int array size
        :return: float adaptive step
        """
        s = 0

        for i in range(0, size):
            s += x[i]**2

        return 1. / s

    def compute_gamma(self, gamma, ii, ii_):
        """ This method calculate gamma
        :param gamma: array for updating
        :param ii: array input image
        :param ii_: array recovered image
        :return: gamma
        """
        for j in range(0, self.p):
            gamma[j] = 0
            for i in range(0, self.n):
                gamma[j] += (ii_[i] - ii[i]) * self.weights_[j, i]

        return gamma

    def training(self):
        """ This method starts training net (calculate weights) using Backpropagation algorithm """
        e = 10
        iteration = 0
        gamma = np.zeros(self.p, dtype=np.float64)

        while e >= 0.01:
            e = 0
            iteration += 1
            start_time = datetime.datetime.now()

            for l in range(0, self.p):
                ii = self.images[l]
                oi = np.dot(ii, self.weights)
                ii_ = np.dot(oi, self.weights_)

                alpha = self.compute_adaptive_step(ii, self.n)
                alpha_ = self.compute_adaptive_step(oi, self.p)
                gamma = self.compute_gamma(gamma, ii, ii_)

                self.compute_weights(ii, oi, ii_, gamma, alpha, alpha_)

            for l in range(0, self.p):
                ii = self.images[l]
                oi = np.dot(ii, self.weights)
                ii_ = np.dot(oi, self.weights_)

                e += self.get_error(ii, ii_)

            delta_time = datetime.datetime.now() - start_time

            print iteration, 'iteration, error =', e, 'took', delta_time.microseconds / 1000., 'ms'

    def process(self):
        self.training(self.images)

        exit(0)
