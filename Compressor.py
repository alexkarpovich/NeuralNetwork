from NeuralNetwork import NeuralNetwork
from scipy import misc
import numpy as np


class Compressor:
    def __init__(self, file_path, wnd_width, wnd_height, p, min_error):
        self.picture = misc.imread(file_path)
        self.p = p
        self.min_error = min_error
        self.width, self.height, self.channels = self.picture.shape
        self.wnd_width, self.wnd_height = self.check_wnd_size(wnd_width, wnd_height)
        self.image_size = self.wnd_width * self.wnd_height * self.channels

    def normalize_color(self, window):
        """ This method normalize each channel value to [-1, 1]
        :param window: array - picture block
        :return: one input image
        """
        image = np.ravel(window).astype(np.float64)

        for i in range(0, self.image_size):
            image[i] = 2. * image[i] / 255 - 1

        return image

    def check_wnd_size(self, wnd_width, wnd_height):
        is_changed = False

        if self.width % wnd_width != 0:
            wnd_width += 1
            is_changed = True

        if self.height % wnd_height != 0:
            wnd_height += 1
            is_changed = True

        if is_changed:
            wnd_width, wnd_height = self.check_wnd_size(wnd_width, wnd_height)

        return wnd_width, wnd_height

    def prepare_images(self):
        """ Build array of input images """
        cols = self.width / self.wnd_width
        rows = self.height / self.wnd_height
        inputs = []

        for i in range(0, cols):
            x_pos = i * self.wnd_width
            for j in range(0, rows):
                y_pos = j * self.wnd_height
                inputs.append(self.normalize_color(self.picture[x_pos: x_pos + self.wnd_width,
                                                                y_pos: y_pos + self.wnd_height]))

        return inputs

    def recover_color(self, one_line_image):
        rec_color = np.empty(0, dtype=np.uint8)
        for value in one_line_image:
            cvalue = (value + 1) * 255 / 2
            rec_color = np.append(rec_color, 255 if cvalue > 255 else 0 if cvalue < 0 else cvalue)

        return rec_color

    def recover_window(self, line):
        rec_color = self.recover_color(line)

        out = np.reshape(rec_color, (-1, self.wnd_width, 3))
        return out

    def recover_image(self, rec_images):
        rec_image = np.copy(self.picture)

        cols = self.width / self.wnd_width
        rows = self.height / self.wnd_height
        c = 0

        for i in range(0, cols):
            x_pos = i * self.wnd_width
            for j in range(0, rows):
                y_pos = j * self.wnd_height
                window = self.recover_window(rec_images[c])
                rec_image[x_pos: x_pos + self.wnd_width,
                          y_pos: y_pos + self.wnd_height] = window
                c += 1

        return rec_image

    def process(self):
        print '[ Prepare input images ]'
        inputs = self.prepare_images()

        print '[ Init Network ]'
        network = NeuralNetwork(inputs, self.p, self.image_size, self.min_error)

        print '[ Start training]'
        network.training()
        # network.load_weights()

        print '[ Start recovering picture ]'
        rec_images = network.process()

        rec_picture = self.recover_image(rec_images)

        print '[ Save recoverd image to file ]'
        misc.imsave('images/rec_image.bmp', rec_picture)
