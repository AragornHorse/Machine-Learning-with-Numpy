import numpy as np


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
    import matplotlib.pyplot as plt

    x, y = get_data_set(100, outlier=0.2)

    plt.scatter(x, y)
    plt.show()








