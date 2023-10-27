import numpy as np


class Poly:
    def __init__(self):
        self.k = None
        self.w = None

    def kernel(self, x, k, bias=True):
        if isinstance(k, int):
            if bias:
                k = list(range(k+1))
            else:
                k = list(i + 1 for i in range(k))

        self.k = k

        x = x.reshape([-1])   # n
        X = np.zeros([x.shape[0], len(k)])
        for line, i in enumerate(k):
            X[:, line] = x ** i

        return X

    def ridge_regression(self, x, y, lbd, k=3, bias=True, is_iter=False, max_iter=100, lr=1e-3, newton=False):

        X = self.kernel(x, k, bias)

        y = y.reshape([-1, 1])

        if not is_iter:
            self.w = np.linalg.inv(X.T @ X + np.diag(np.full([X.shape[1]], lbd * X.shape[0]))) @ X.T @ y

            loss = np.mean((y - X @ self.w) ** 2)

        else:

            self.w = np.random.random([X.shape[1], 1])

            loss = []

            for epoch in range(max_iter):

                if not newton:
                    self.w -= lr * (X.T @ (X @ self.w - y) / X.shape[0] + lbd * self.w)
                else:
                    self.w -= np.linalg.inv(X.T @ X / X.shape[0] + np.diag(np.full([X.shape[1]], lbd))) @ \
                              (X.T @ (X @ self.w - y) / X.shape[0] + lbd * self.w)

                loss.append(np.mean((y - X @ self.w) ** 2))

        return loss

    def lasso(self, x, y, lbd, k=3, bias=True, max_iter=500, lr=1e-4, newton=False, coordinate=False):

        X = self.kernel(x, k, bias)

        y = y.reshape([-1, 1])

        self.w = np.zeros([X.shape[1], 1])

        loss = []

        if not coordinate:
            for i in range(max_iter):

                grad = (X.T @ (X @ self.w - y) / X.shape[0] + lbd * np.sign(self.w))

                grad = grad * (1 - (self.w == 0) * (np.abs(grad) <= lbd))

                if not newton:
                    self.w -= lr * grad
                    # self.w -= lr * (X.T @ (X @ self.w - y) / X.shape[0] + lbd * np.sign(self.w))
                else:
                    self.w -= np.linalg.inv(X.T @ X / X.shape[0]) @ (X.T @ (X @ self.w - y) / X.shape[0] + lbd * np.sign(self.w))

                loss.append(np.mean((y - X @ self.w) ** 2))
        else:

            for epoch in range(max_iter // X.shape[1]):

                for k in range(X.shape[1]):
                    wk_ = self.w[[j for j in range(X.shape[1]) if j != k]].reshape([-1, 1])

                    xk = X[:, k].reshape([-1, 1])
                    xk_ = X[:, [j for j in range(X.shape[1]) if j != k]]

                    ak = wk_.T @ xk_.T @ xk - xk.T @ y
                    bk = xk.T @ xk
                    ck = x.shape[0] * lbd

                    if ak < -ck:
                        self.w[k, 0] = - ((ak + ck) / bk)
                    elif ak > -ck:
                        self.w[k, 0] = - ((ak - ck) / bk)
                    else:
                        self.w[k, 0] = 0

                loss.append(np.mean((y - X @ self.w) ** 2))

        return loss

    def ransac(self, x, y, k, bias=True, batch_size=10, outer=0.1, accept=0.9, out_thres=0.1):

        import random

        max_iter = np.log(1 - accept) / (np.log(1 - (1 - outer) ** batch_size + 1e-30) + 1e-30)

        max_iter = int(max_iter)

        best_w = None
        best_score = None

        idxs = list(range(x.shape[0]))

        for i in range(max_iter):
            idx = random.sample(idxs, batch_size)

            self.ridge_regression(x[idx], y[idx], lbd=0, k=k, bias=bias, is_iter=False)

            losses = np.abs(y[idx] - self.kernel(x[idx], k, bias) @ self.w)
            out_num = np.mean(losses > out_thres)

            if best_score is None:
                best_w = self.w
                best_score = out_num

            elif out_num < best_score:
                best_w = self.w
                best_score = out_num

        self.w = best_w
        return best_score

    def __call__(self, x):
        x = x.reshape([-1])
        X = self.kernel(x, self.k)
        return (X @ self.w).reshape([-1])


if __name__ == '__main__':
    model = Poly()
    x = np.array(range(10))
    # print(model.kernel(x, 4))

    import data

    x, y = data.get_data_set(100, outlier=0., outer_miu=3)

    print(model.ridge_regression(x, y, 0.3, k=3, is_iter=False, newton=True))

    # print(model.lasso(x, y, 0.05, 4, newton=False))

    import matplotlib.pyplot as plt

    plt.scatter(x, y)
    #
    # x = np.linspace(-4, 4, 200)
    # y_hat = model.kernel(x, model.k) @ model.w
    # plt.plot(x, y_hat)
    # plt.show()
    #
    # print(model.w)
    # x, y = data.get_data_set(100, outlier=0.2, outer_miu=3)
    # plt.scatter(x, y)
    #
    # print(model.ransac(x, y, 3, True, batch_size=20, outer=0.2, accept=0.999))
    #
    # x = np.linspace(-4, 4, 200)
    # y_hat = model.kernel(x, model.k) @ model.w
    # plt.plot(x, y_hat)
    # plt.show()

    print(model.lasso(x, y, 1., 10, newton=False, lr=1e-3, max_iter=1000, coordinate=True))
    # print(model.ridge_regression(x, y, 0., k=3, is_iter=True, newton=False))

    x = np.linspace(-4, 4, 200)
    y_hat = model.kernel(x, model.k) @ model.w
    plt.plot(x, y_hat)
    plt.show()

    print(model.w)













