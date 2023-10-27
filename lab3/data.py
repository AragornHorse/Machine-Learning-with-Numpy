import numpy as np


def get_data(means, covs, nums):
    xs = []
    ys = []
    for i in range(means.shape[0]):
        xs.append(np.random.multivariate_normal(means[i], covs[i], size=nums[i]))
        ys.append(np.full([nums[i], 1], i))

    xs = np.concatenate(xs, axis=0)
    ys = np.concatenate(ys, axis=0).reshape([-1])

    return xs, ys


if __name__ == '__main__':
    x = get_data(np.array([[1, 1], [2, 2]]), np.random.randn(2, 2, 2), [10, 20])
    import matplotlib.pyplot as plt
    plt.scatter(x[:, 0], x[:, 1])
    plt.show()


