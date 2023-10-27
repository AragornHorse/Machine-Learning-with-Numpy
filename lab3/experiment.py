import cluster
import data
import numpy as np
import matplotlib.pyplot as plt


def kmeans():

    means = np.array([
        [0., 0.],
        [1., 1.],
        [1., 0.]
    ])

    covs = np.array([
        [
            [0.1, -0.1],
            [-0.1, 0.2]
        ],
        [
            [0.05, 0.04],
            [0.04, 0.05]
        ],
        [
            [0.03, -0.026],
            [-0.026, 0.03]
        ]
    ])

    nums = [30, 40, 20]

    x, y = data.get_data(means, covs, nums)

    model = cluster.KMeans()

    y_hat = model.fit(x, 3, max_iter=500)

    plt.scatter(x[:, 0], x[:, 1], c=y_hat, s=10.)
    plt.show()


kmeans()

def GMM():

    means = np.array([
        [0., 0.],
        [1., 1.],
        [0.7, 0.]
    ])

    covs = np.array([
        [
            [0.1, 0],
            [0, 0.1]
        ],
        [
            [0.03, 0.],
            [0., 0.03]
        ],
        [
            [0.03, 0.0],
            [0., 0.03]
        ]
    ])

    nums = [30, 40, 20]

    x, y = data.get_data(means, covs, nums)

    model = cluster.GMM()

    ls = model.EM(x, 3, max_iter=500)

    plt.scatter(x[:, 0], x[:, 1], c=model.cls, s=10.)
    plt.show()


GMM()

def GMM_lc():
    plt.figure(figsize=(12, 6))
    for _ in range(10):
        means = np.array([
            [0., 0.],
            [1., 1.],
            [0.7, 0.]
        ]) + np.random.random([3, 2]) * 0.1

        covs = np.array([
            [
                [0.1, 0],
                [0, 0.1]
            ],
            [
                [0.03, 0.],
                [0., 0.03]
            ],
            [
                [0.03, 0.0],
                [0., 0.03]
            ]
        ]) + np.random.random([3, 2, 2]) * 0.005

        nums = [30, 40, 20]

        x, y = data.get_data(means, covs, nums)

        model = cluster.GMM()

        ls = model.EM(x, 3, max_iter=500)

        plt.plot(ls, linewidth=0.5)
    plt.show()

GMM_lc()


def ic():
    means = np.array([
        [0., 0.],
        [1., 1.],
        [1., 0.],
        [0., 1.],
        [1., 2.]
    ])

    covs = np.array([
        [
            [0.1, 0],
            [0, 0.1]
        ],
        [
            [0.03, 0.],
            [0., 0.03]
        ],
        [
            [0.02, 0.0],
            [0., 0.2]
        ],
        [
            [0.01, 0.],
            [0., 0.1]
        ],
        [
            [0.01, 0.],
            [0., 0.03]
        ]
    ])

    nums = [30, 40, 20, 20, 30]

    x, y = data.get_data(means, covs, nums)

    model = cluster.GMM()

    ls = cluster.ic(x, model, [1, 2, 3, 4, 5, 6], cluster.aic)

    plt.plot([1, 2, 3, 4, 5, 6], ls, linewidth=0.5)

    ls = cluster.ic(x, model, [1, 2, 3, 4, 5, 6], cluster.bic)

    plt.plot([1, 2, 3, 4, 5, 6], ls, linewidth=0.5)

    plt.legend(['AIC', 'BIC'])

    plt.show()

ic()

def gmm_cov():
    means = np.array([
        [0., 0.],
        [1., 1.],
        [1., 0.],
        [0., 1.]
    ])

    covs = np.array([
        [
            [0.1, 0.01],
            [0.01, 0.1]
        ],
        [
            [0.03, 0.],
            [0., 0.03]
        ],
        [
            [0.02, 0.0],
            [0., 0.2]
        ],
        [
            [0.01, 0.],
            [0., 0.1]
        ]
    ])

    nums = [30, 40, 30, 50]

    x, y = data.get_data(means, covs, nums)

    model = cluster.GMM()

    model.EM(x, 4, 200)

    print(model.mius)

    print(model.sigmas)

    print(model.pi)

gmm_cov()


def app():
    from sklearn.datasets import load_iris

    data = load_iris()

    x = data['data']
    y = data['target']

    # cov = (x.T @ x) / x.shape[0]
    #
    # U, D, _ = np.linalg.svd(cov)
    #
    # D = np.diag(D ** -0.5)
    #
    # print(D.shape, U.shape, x.shape)

    # x = (D @ U.T @ x.T).T

    # print(x.T @ x)

    x = (x - np.mean(x, axis=0, keepdims=True)) / np.std(x, axis=0, keepdims=True)

    # pca = cluster.PCA()
    # pca.get_w(x, 2)
    # x = pca(x)
    #
    # plt.scatter(x[:, 0], x[:, 1])
    # plt.show()

    model = cluster.KMeans()

    y_hat = model.fit(x, k=3, max_iter=5, init=True)

    plt.scatter(x[:, 0], x[:, 1], c=y_hat)
    plt.show()

    print(y)
    print(y_hat)

    print(cluster.accuracy_metric(y_hat, y))

    model = cluster.GMM()

    model.EM(x, 3, 1000)

    y_hat = model.cls
    plt.scatter(x[:, 0], x[:, 1], c=y_hat)
    plt.show()

    print(y)
    print(y_hat)

    print(cluster.accuracy_metric(y_hat, y))

app()

# import sklearn
# print(sklearn.__version__)


