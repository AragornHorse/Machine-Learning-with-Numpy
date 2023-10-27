import numpy as np
import random


train_x = np.load(r"./minst/data/train_x.npy")
train_y = np.load(r"./minst/data/train_y.npy")
test_x = np.load(r"./minst/data/test_x.npy")
test_y = np.load(r"./minst/data/test_y.npy")


class DataLoader:
    def __init__(self, x, y, batch_size):
        self.x = x
        self.y = y
        self.batch_size = batch_size

    def __getitem__(self, item):
        idxs = random.sample(range(self.x.shape[0]), k=self.batch_size)
        return self.x[idxs], self.y[idxs]

    def __len__(self):
        return self.x.shape[0] // self.batch_size


if __name__ == '__main__':

    loader = DataLoader(train_x, train_y, 32)

    for data in loader:
        x, y = data
        print(x.shape, y.shape)