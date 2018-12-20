# encode:utf-8
import numpy as np
from matplotlib import pyplot as plt, animation as anim


def get_datas():
    return np.array([[3, 3], [4, 3], [1, 1]]), np.array([1, 1, -1])


class Model():
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.w = None
        self.features = None
        self.labels = None
        self.gram = None

        self.line = 0
        self.index = 0
        self.update_count = 0

    def cal_gram(self):
        l = self.features.shape[0]
        self.gram = np.zeros((l, l))
        for i in range(l):
            for j in range(l):
                self.gram[i][j] = np.dot(self.features[i], self.features[j])

    def check_step(self, i):
        loss = self.labels[i] * (np.dot(self.a * self.labels, self.gram[i]) + self.b)
        print("check_step ", i, loss)
        if loss <= 0:
            return False
        return True

    def check(self):
        l = self.features.shape[0]
        for i in range(l):
            if not self.check_step(i):
                return False
        return True

    def step(self, index):
        self.a[index] = self.a[index] + 1
        self.b = self.b + self.labels[index]

    def train(self, features, labels):
        self.features = features
        self.labels = labels
        self.cal_gram()
        for i in range(10):
            print("epoch ", i)
            if self.check():
                print("finish")
                break
            for i in range(self.features.shape[0]):
                if not self.check_step(i):
                    self.step(i)
                    print("train ", i, self.a, self.b)

    def get_x1(self, x):
        self.w = np.dot(self.a * self.labels, self.features)
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
            if not self.check_step(i):
                self.step(i)
                break
        if self.index == self.features.shape[0] - 1:
            print("finish one epoch")
            self.index = 0
        self.drawLine()

        # if self.check():
        #     print("finish train ", self.w, self.b)
        #     self.animFunc.event_source.stop()

    def draw_update(self, point):
        self.update_count += 1
        if self.update_count % 5 == 0:
            self.draw_step()

    def draw(self, features, labels):
        self.features = features
        self.labels = labels
        self.cal_gram()

        fig = plt.figure()
        axes1 = fig.add_subplot(111)
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
        self.cal_gram()

        fig = plt.figure()
        axes1 = fig.add_subplot(111)
        axes1.set_xlim(0, 6)
        axes1.set_ylim(0, 6)
        axes1.plot(features[:, 0], features[:, 1], ".", color='red', animated=False)

        self.line, = axes1.plot([], [], "-", color='red', animated=False)

        self.animFunc = anim.FuncAnimation(fig, self.draw_update_by_button, repeat=True)
        fig.canvas.mpl_connect('key_press_event', self.key_press)
        plt.show()


if __name__ == '__main__':
    features, labels = get_datas()
    a = np.zeros((3))
    b = 0

    model = Model(a, b)
    # model.train(features, labels)
    # model.draw(features, labels)
    model.draw_by_button(features, labels)
