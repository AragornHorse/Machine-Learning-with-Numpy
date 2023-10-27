import numpy as np


def get_init_mean(x, k):

    def min_dis(xs, cs):
        dis = []
        for i in range(xs.shape[0]):
            x_ = xs[i]
            dis_ = np.sum((x_.reshape([1, -1]) - cs) ** 2, axis=-1)
            dis.append(np.min(dis_))

        return np.array(dis)

    means = [x[0]]
    for _ in range(k - 1):
        dis = min_dis(x, np.array(means))
        idx = np.argmax(dis)
        means.append(x[idx])

    return np.array(means)


def gaussian(x, mean, cov):
    n, d = x.shape
    dis = x - mean.reshape([1, -1])

    dis = dis.reshape([n, 1, d]) @ np.linalg.inv(cov) @ dis.reshape([n, d, 1])
    det = np.linalg.det(cov)
    norm = 1 / ((2 * np.pi) ** (d / 2) * det ** 0.5)
    return norm * np.exp(-0.5 * dis.reshape([-1]))


class KMeans:
    def __init__(self):
        self.centers = []
        self.cls = []
        self.k = None

    def fit(self, x, k, max_iter=100, init=False):
        self.k = k
        n, h = x.shape
        if init:
            self.centers = get_init_mean(x, k)   # k, h
        else:
            self.centers = np.random.randn(k, x.shape[1])
        dis_ = np.repeat(x.T.reshape(1, h, n), k, axis=0)   # k, h, n

        for epoch in range(max_iter):
            dis = np.sum((self.centers.reshape([k, h, 1]) - dis_) ** 2, axis=1).T   # n, k
            self.cls = np.argmax(dis, axis=1)   # n
            # print(self.cls)

            for k_ in range(k):
                idx = np.nonzero(self.cls == k_)[0]
                if idx.shape[0] == 0:
                    self.centers[k_] = np.random.random([x.shape[1]])
                    continue
                x_ = x[idx, :]
                self.centers[k_] = np.mean(x_, axis=0)
        return self.cls

    def __call__(self, x):
        n, h = x.shape
        dis_ = np.repeat(x.T.reshape(1, h, n), self.k, axis=0)
        dis = np.sum((self.centers.reshape([self.k, h, 1]) - dis_) ** 2, axis=1).T
        self.cls = np.argmax(dis, axis=1)
        return self.cls


class GMM:
    def __init__(self):
        self.mius = []
        self.sigmas = []
        self.pi = []
        self.cls = []
        self.k = None
        self.gammas = []

    def gaussian(self, x, mean, cov):
        n, d = x.shape
        dis = x - mean.reshape([1, -1])

        dis = dis.reshape([n, 1, d]) @ np.linalg.inv(cov) @ dis.reshape([n, d, 1])
        det = np.linalg.det(cov)
        norm = 1 / ((2 * np.pi) ** (d / 2) * det ** 0.5)
        return norm * np.exp(-0.5 * dis.reshape([-1]))

    def EM(self, x, k, max_iter=100):
        n, h = x.shape
        # self.mius = np.array([x[i] for i in range(k)])  # k, h
        self.mius = get_init_mean(x, k)
        self.sigmas = np.array([np.eye(h) for _ in range(k)])  # k, h, h
        self.pi = np.ones([k]) / k  # k
        self.k = k
        self.gammas = np.zeros([n, k])

        ls = []

        for epoch in range(max_iter):

            p_cls_x = np.zeros([n, k])

            for i in range(k):
                p_cls_x[:, i] = self.gaussian(x, self.mius[i], self.sigmas[i])

            self.gammas = p_cls_x / (np.sum(p_cls_x, axis=-1, keepdims=True))

            for i in range(k):
                norm = np.sum(self.gammas[:, i])
                self.mius[i] = (self.gammas[:, i].reshape([1, n]) @ x).reshape([-1]) / norm
                dis = x - self.mius[i].reshape([1, -1])   # n, h
                self.sigmas[i] = (dis * self.gammas[:, i].reshape([-1, 1])).T @ dis / norm

            self.pi = np.mean(self.gammas, axis=0)
            self.cls = np.argmax(self.gammas, axis=-1)

            ls.append(log_likelihood(self, x))

        return ls

    def __call__(self, x):
        k = self.k
        n, h = x.shape

        p_cls_x = np.zeros([n, k])

        for i in range(k):
            p_cls_x[:, i] = self.gaussian(x, self.mius[i], self.sigmas[i])

        gammas = p_cls_x / np.sum(p_cls_x, axis=-1, keepdims=True)

        cls = np.argmax(gammas, axis=-1)

        return cls, gammas


def log_likelihood(model, xs):
    n = xs.shape[0]

    k = model.k

    p_cls_x = np.zeros([n, k])

    for i in range(k):
        p_cls_x[:, i] = gaussian(xs, model.mius[i], model.sigmas[i])

    pi = model.pi.reshape([-1, 1])
    p_x = p_cls_x @ pi
    return np.sum(np.log(p_x + 1e-30))


def aic(x, model, k):
    k = k + k * x.shape[1] + k * x.shape[1] * (x.shape[1] - 1) / 2
    l = log_likelihood(model, x)
    return - 2 * l + 2 * k


def bic(x, model, k):
    k = k + k * x.shape[1] + k * x.shape[1] * (x.shape[1] - 1) / 2
    l = log_likelihood(model, x)
    return - 2 * l + k * np.log(x.shape[0])


def ic(x, model, ks, ic_func):
    scores = []

    for k in ks:
        model.EM(x, k, max_iter=1000)
        scores.append(ic_func(x, model, k))

    return scores


def accuracy_metric(y_hat, y):
    num = np.zeros([np.max(y_hat) + 1, np.max(y) + 1])

    for i, y_ in enumerate(y):
        y_hat_ = y_hat[i]
        num[y_hat_, y_] += 1

    num = num / np.sum(num)

    return num


class PCA:
    def __init__(self):
        self.w = None
        self.lbd = None

    def get_w(self, x, k=4):
        x = np.copy(x)  # b, h
        x = x - np.mean(x, axis=0).reshape(1, -1)
        cov = np.cov(x, rowvar=False)   # h, h
        lbd, vec = np.linalg.eig(cov)
        idx = np.argsort(lbd)[-k:]
        w = vec[:, idx]  # h, k
        self.w = w
        self.lbd = np.sort(lbd)
        return w

    def __call__(self, x, k=5, update_w=True):
        if self.w is None or self.w.shape[1] != k:
            update_w = True
        if update_w:
            self.get_w(x, k)
        return np.dot(x, self.w)


if __name__ == '__main__':
    # model = KMeans()
    import matplotlib.pyplot as plt

    x = np.concatenate([np.random.randn(100, 2) * 2, np.random.randn(100, 2) + 3, np.random.randn(100, 2) * 0.3 -3], 0)

    #
    # model.fit(x, 3)
    #
    # plt.scatter(x[:, 0], x[:, 1], c=model.cls)
    #
    # plt.show()
    model = GMM()
    model.EM(x, 3, max_iter=1000)
    x = np.concatenate([np.random.randn(100, 2) * 2, np.random.randn(100, 2) + 3, np.random.randn(100, 2) * 0.3 - 3], 0)
    #
    # model.fit(x, 3)
    #
    # plt.scatter(x[:, 0], x[:, 1], c=model.cls)
    #
    # plt.show()
    cls, _ = model(x)

    plt.scatter(x[:, 0], x[:, 1], c=cls)
    means = get_init_mean(x, 3)

    plt.scatter(means[:, 0], means[:, 1], s=100)

    plt.show()

    print(model.mius)
    print(model.sigmas)
    # from scipy.stats import multivariate_normal
    # print(multivariate_normal.pdf(np.zeros([1, 2]), mean=np.zeros([2]), cov=np.eye(2) / 1e4))

    print(ic(x, model, ks=[1, 2, 3, 4, 5, 6, 7], ic_func=bic))











