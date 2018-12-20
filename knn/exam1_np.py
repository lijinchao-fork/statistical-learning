# encode:utf-8
import numpy as np
from anytree import Node, RenderTree


def get_datas():
    return np.array([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]])


class KDNode(Node):
    def __init__(self, feature, parentNode):
        self.feature = feature
        self.visited = False
        name = "({},{})".format(feature[0], feature[1])
        super(KDNode, self).__init__(name, parent=parentNode)


class Model():
    def __init__(self, features):
        self.features = features
        self.dims = len(features.shape)
        self.root = None
        self.result_len = 0
        self.result = []
        self.target = None

        self.nodes = []

    def build_kd(self, root, childs, dimIndex):
        if len(childs) == 0:
            return
        # childs.sort(order=str(dimIndex))
        childs = sorted(childs, key=lambda x: x[dimIndex])
        childs = np.array(childs)
        half = childs.shape[0] / 2
        print("childs ", childs)
        curRoot = KDNode(childs[half], root)
        if root == None:
            self.root = curRoot
        lefts = childs[:half]
        rights = childs[half + 1:]
        self.build_kd(curRoot, lefts, (dimIndex + 1) % self.dims)
        self.build_kd(curRoot, rights, (dimIndex + 1) % self.dims)

    def build(self):
        status = 0
        curnode = None
        for feature in self.features:
            if curnode == None:
                curnode = KDNode(feature, None)
                self.root = curnode
            else:
                node = KDNode(feature, curnode)
                status += 1
                if status >= 2:
                    curnode = node

    def draw(self):
        for pre, fill, node in RenderTree(self.root):
            print("%s%s" % (pre, node.name))

    def cal_dis(self, a, b):
        return np.linalg.norm(a - b)

    def find_leaf(self, node, target, dimIndex):
        if len(node.children) == 0:
            return [node]
        else:
            if target[dimIndex] < node.feature[dimIndex]:
                return [node] + self.find_leaf(node.children[0], target, (dimIndex + 1) % self.dims)
            else:
                if len(node.children) == 2:
                    return [node] + self.find_leaf(node.children[1], target, (dimIndex + 1) % self.dims)
                else:
                    return [node]

    def find_one(self, node, target, dim_index):
        pass


if __name__ == '__main__':
    features = get_datas()
    target = np.array((6, 2))

    model = Model(features)
    # model.build()
    model.build_kd(None, features, 0)
    model.draw()

    # nodes = model.find_leaf(model.root, target, 0)
    # print nodes

    model.find_one(model.root, target, 0)
