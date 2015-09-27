from window import Window
from scipy import misc


class Compressor:
    def __init__(self, file_path, wnd_width, wnd_height, p):
        self.image = misc.imread(file_path)
        self.p = p
        self.width, self.height, _ = self.image.shape
        self.wnd_width, self.wnd_height = self.check_wnd_size(wnd_width, wnd_height)

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

    def process(self):
        cols = self.width / self.wnd_width
        rows = self.height / self.wnd_height

        for i in range(0, cols):
            x_pos = i * self.wnd_width
            for j in range(0, rows):
                y_pos = j * self.wnd_height
                w = Window(self.image[x_pos: x_pos + self.wnd_width, y_pos: y_pos + self.wnd_height],
                           self.p,
                           self.wnd_width,
                           self.wnd_height)

                w.process()
