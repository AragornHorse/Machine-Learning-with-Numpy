import numpy as np
import random

train_x = np.load(r"./minst/data/train_x.npy")
train_y = np.load(r"./minst/data/train_y.npy")
test_x = np.load(r"./minst/data/test_x.npy")
test_y = np.load(r"./minst/data/test_y.npy")

np.random.seed(2)
random.seed(7)


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


def get_data_set(
        n, miu=0., sigma=0.1, outlier=0., func=np.sin, shuffle=False, x_range=None, outer_miu=1., outer_sigma=1.
):
    """
        generate dataset
    :param outer_sigma: std of outlier
    :param outer_miu:   expression of outlier
    :param n:           number of sample
    :param miu:         expression of gaussian noise
    :param sigma:       std of gaussian noise
    :param outlier:     outlier rate
    :param func:        true function
    :param shuffle:     whether x is in order
    :param x_range:     define domain of x
    :return: x, y
    """

    if x_range is None:
        x_range = [-4, 4]

    x = np.random.random([n]) * (x_range[1] - x_range[0]) + x_range[0]

    if not shuffle:
        x = np.sort(x)

    y = func(x)

    noise = np.random.randn(x.shape[0]) * sigma + miu

    y = y + noise

    if outlier > 0.:
        is_out = (np.random.random([x.shape[0]]) < outlier).astype(int)
        y = y * (1 - is_out) + is_out * (np.random.randn(y.shape[0]) * outer_sigma + outer_miu)

    return x, y


if __name__ == '__main__':

    loader = DataLoader(train_x, train_y, 32)

    for data in loader:
        x, y = data
        print(x.shape, y.shape)
