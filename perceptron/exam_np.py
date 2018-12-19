# encode:utf-8
import numpy as np
from matplotlib import pyplot as plt, animation as anim


def get_datas():
    return np.array([[3, 3], [4, 3], [1, 1]]), np.array([[1], [1], [-1]])


class Model():
    def __init__(self, w, b):
        self.w = w
        self.b = b
        self.line = 0
        self.index = 0
        self.update_count = 0
        self.features = None
        self.labels = None

    def check(self, features, labels):
        result = labels * (np.dot(features, self.w) + self.b)
        return (result > 0).all()

    def step(self, feature, label):
        self.w = self.w + np.expand_dims(feature, 1) * label
        self.b = self.b + label

    def train(self, features, labels):
        while not self.check(features, labels):
            for i in range(features.shape[0]):
                if self.check(features[i], labels[i]):
                    continue
                self.step(features[i], labels[i])
                print("train ", self.w, self.b)

    def get_x1(self, x):
        w1 = self.w[1] if self.w[1] != 0 else 0.0000001
        x1 = -self.b - self.w[0] * x / w1
        return x1

    def drawLine(self):
        x = [0, 6]
        y = [self.get_x1(0), self.get_x1(6)]
        self.line.set_data(x, y)

    def draw_step(self):
        for i in range(self.index, self.features.shape[0]):
            self.index = i
            if not self.check(self.features[i], self.labels[i]):
                self.step(self.features[i], self.labels[i])
                print self.w, self.b
                break
        if self.index == self.features.shape[0] - 1:
            print("finish one epoch")
            self.index = 0
        self.drawLine()
        # if self.check(self.features, self.labels):
        #     print("finish train ", self.w, self.b)
        #     self.animFunc.event_source.stop()

    def draw_update(self, point):
        self.update_count += 1
        if self.update_count % 5 == 0:
            self.draw_step()

    def draw(self, features, labels):
        self.features = features
        self.labels = labels

        fig = plt.figure()
        axes1 = fig.add_subplot(111)
        # line = axes1.scatter(points[:, 0], points[:, 1], 1)
        axes1.set_xlim(0, 6)
        axes1.set_ylim(0, 6)
        axes1.plot(features[:, 0], features[:, 1], ".", color='red', animated=False)

        self.line, = axes1.plot([], [], "-", color='red', animated=False)

        self.animFunc = anim.FuncAnimation(fig, self.draw_update, repeat=True)
        plt.show()

    def draw_update_by_button(self, point):
        self.drawLine()

    def key_press(self, event):
        if event.key == ' ':
            self.draw_step()

    def draw_by_button(self, features, labels):
        self.features = features
        self.labels = labels

        fig = plt.figure()
        axes1 = fig.add_subplot(111)
        # line = axes1.scatter(points[:, 0], points[:, 1], 1)
        axes1.set_xlim(0, 6)
        axes1.set_ylim(0, 6)
        axes1.plot(features[:, 0], features[:, 1], ".", color='red', animated=False)

        self.line, = axes1.plot([], [], "-", color='red', animated=False)

        self.animFunc = anim.FuncAnimation(fig, self.draw_update_by_button, repeat=True)
        fig.canvas.mpl_connect('key_press_event', self.key_press)

        plt.show()


if __name__ == '__main__':
    features, labels = get_datas()
    w = np.zeros((2, 1))
    b = 0

    model = Model(w, b)
    # model.train(features, labels)
    # model.draw(features, labels)
    model.draw_by_button(features, labels)
