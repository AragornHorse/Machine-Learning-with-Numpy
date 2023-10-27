import data
import poly_regression
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


# print(matplotlib._get_version())


def visual_data():
    x, y = data.get_data_set(
        100, 0., 0.1, 0.5, shuffle=True, outer_miu=1., outer_sigma=1.
    )

    plt.scatter(x, y, s=5., c=[[0.2, 0.2, 0.6] for _ in range(x.shape[0])])
    plt.show()


# visual_data()


def ridge_lbds():

    noise = 0.1
    num = 100
    lbds = [0., 1e-3, 1e-2, 1e-1, 2e-1, 3e-1, 4e-1, 5e-1]

    x, y = data.get_data_set(
        num, miu=0., sigma=noise, outlier=0., shuffle=True, outer_miu=1., outer_sigma=1.
    )

    plt.scatter(x, y, s=2., c=[[0.6, 0.2, 0.2] for _ in range(x.shape[0])])

    model = poly_regression.Poly()

    for lbd in lbds:
        model.ridge_regression(x, y, lbd=lbd, k=3, is_iter=True, newton=True, max_iter=900, lr=3e-4)

        y_hat = model(np.linspace(-4.3, 4.3, 100))

        plt.plot(np.linspace(-4.3, 4.3, 100), y_hat, linewidth=0.5)

    plt.legend(["data"] + ["lbd={}".format(lbd) for lbd in lbds], prop={'size': 8})
    plt.show()


# ridge_lbds()

def learning_curve():
    noise = 0.1
    num = 100
    lbds = [0., 1e-3, 1e-2, 1e-1, 2e-1, 3e-1]

    x, y = data.get_data_set(
        num, miu=0., sigma=noise, outlier=0., shuffle=True, outer_miu=1., outer_sigma=1.
    )

    model = poly_regression.Poly()
    plt.figure(figsize=(12, 6))

    for lbd in lbds:
        loss = model.ridge_regression(x, y, lbd=lbd, k=3, is_iter=True, newton=False, max_iter=100, lr=3e-4)

        plt.plot(loss, linewidth=0.5)

    for lbd in lbds:
        loss = model.ridge_regression(x, y, lbd=lbd, k=3, is_iter=True, newton=True, max_iter=100, lr=3e-4)

        plt.plot(loss, linewidth=0.5, linestyle='--')

    plt.legend(["lbd={}, grad".format(lbd) for lbd in lbds] + ["lbd={}, newton".format(lbd) for lbd in lbds], prop={'size': 8})
    plt.show()


# learning_curve()

def lasso_lbd():

    noise = 0.1
    num = 100
    lbds = [0., 1e-3, 1e-2, 1e-1, 2e-1, 3e-1, 4e-1, 5e-1]

    x, y = data.get_data_set(
        num, miu=0., sigma=noise, outlier=0., shuffle=True, outer_miu=1., outer_sigma=1.
    )

    plt.scatter(x, y, s=2., c=[[0.6, 0.2, 0.2] for _ in range(x.shape[0])])

    model = poly_regression.Poly()

    for lbd in lbds:
        model.lasso(x, y, lbd=lbd, k=3, newton=False, max_iter=900, lr=1e-3, coordinate=True)

        y_hat = model(np.linspace(-4.3, 4.3, 100))

        plt.plot(np.linspace(-4.3, 4.3, 100), y_hat, linewidth=0.5)

    plt.legend(["data"] + ["lbd={}".format(lbd) for lbd in lbds], prop={'size': 8})
    plt.show()


# lasso_lbd()

def lasso_w():

    noise = 0.1
    num = 100
    lbds = [0., 1e-3, 1e-2, 1e-1, 2e-1, 3e-1, 4e-1, 5e-1, 8e-1, 1.]

    x, y = data.get_data_set(
        num, miu=0., sigma=noise, outlier=0., shuffle=True, outer_miu=1., outer_sigma=1.
    )

    # plt.scatter(x, y, s=2., c=[[0.6, 0.2, 0.2] for _ in range(x.shape[0])])

    model = poly_regression.Poly()

    ws = []

    for lbd in lbds:
        model.lasso(x, y, lbd=lbd, k=4, newton=False, coordinate=False, max_iter=2000, lr=1e-4)

        ws.append(model.w.reshape([-1]))

    ws = np.array(ws).T

    for l in ws:
        plt.plot(lbds, l, linewidth=1.)

    plt.legend(['w{}'.format(i) for i in range(ws.shape[0])])
    plt.show()


# lasso_w()


def outlier():
    noise = 0.1
    num = 100
    lbds = [0., 1e-3, 1e-2, 1e-1, 2e-1, 3e-1]

    x, y = data.get_data_set(
        num, miu=0., sigma=noise, outlier=0.1, shuffle=True, outer_miu=2., outer_sigma=1.
    )

    plt.scatter(x, y, s=2., c=[[0.6, 0.2, 0.2] for _ in range(x.shape[0])])

    model = poly_regression.Poly()

    for lbd in lbds:
        model.ridge_regression(x, y, lbd=lbd, k=3, is_iter=True, newton=True, max_iter=900, lr=3e-4)

        y_hat = model(np.linspace(-4.3, 4.3, 100))

        plt.plot(np.linspace(-4.3, 4.3, 100), y_hat, linewidth=0.5)

    for lbd in lbds:
        model.lasso(x, y, lbd=lbd, k=3, newton=True, max_iter=900, lr=1e-3)

        y_hat = model(np.linspace(-4.3, 4.3, 100))

        plt.plot(np.linspace(-4.3, 4.3, 100), y_hat, linewidth=0.5)

    model.ransac(x, y, k=3, batch_size=10, outer=0.5, accept=0.99)

    y_hat = model(np.linspace(-4.3, 4.3, 100))

    plt.plot(np.linspace(-4.3, 4.3, 100), y_hat, linewidth=0.5)

    plt.legend(
        ["data"] + ["lbd={}, ridge".format(lbd) for lbd in lbds] + ["lbd={}, lasso".format(lbd) for lbd in lbds] + ["RANSAC"],
        prop={'size': 8}
    )
    plt.show()


# outlier()

def over_fit():

    def mse(y_hat, y):
        return np.mean((y_hat - y) ** 2)

    def explained_variance_score(y_hat, y):
        return 1 - np.var(y - y_hat) / np.var(y)

    def max_error(y_hat, y):
        return np.max(np.abs(y - y_hat))

    def mae(y_hat, y):
        return np.mean(np.abs(y - y_hat))

    def median_ae(y_hat, y):
        return np.median(np.abs(y - y_hat))

    def r2(y_hat, y):
        return 1 - mse(y_hat, y) / np.var(y)

    def all_score(y_hat, y):
        return [
            mse(y_hat, y), explained_variance_score(y_hat, y),
            max_error(y_hat, y), mae(y_hat, y), median_ae(y_hat, y), r2(y_hat, y)
        ]

    noise = 0.2
    num = 10
    ks = [1, 2, 3, 4, 5]

    lbd = 0.

    x_train, y_train = data.get_data_set(
        num, miu=0., sigma=noise, outlier=0., shuffle=True, outer_miu=1., outer_sigma=1.
    )
    plt.scatter(x_train, y_train, s=2., c=[[0.6, 0.2, 0.2] for _ in range(x_train.shape[0])])

    x_test, y_test = data.get_data_set(
        num, miu=0., sigma=noise, outlier=0., shuffle=True, outer_miu=1., outer_sigma=1.
    )
    plt.scatter(x_test, y_test, s=2., c=[[0.1, 0.6, 0.2] for _ in range(x_train.shape[0])])

    model = poly_regression.Poly()

    for k in ks:
        model.ridge_regression(x_train, y_train, lbd, k=k)   # increase k

        print(all_score(model(x_train), y_train))

        plt.plot(np.linspace(-4.3, 4.3, 100), model(np.linspace(-4.3, 4.3, 100)), linewidth=0.5)

        y_hat = model(x_test)

        print(all_score(y_hat, y_test))

        print("")

    plt.legend(['train_set', 'test_set'] + ["k={}".format(k) for k in ks])
    plt.show()

# over_fit()


def over_fit_lbd():

    def mse(y_hat, y):
        return np.mean((y_hat - y) ** 2)

    def explained_variance_score(y_hat, y):
        return 1 - np.var(y - y_hat) / np.var(y)

    def max_error(y_hat, y):
        return np.max(np.abs(y - y_hat))

    def mae(y_hat, y):
        return np.mean(np.abs(y - y_hat))

    def median_ae(y_hat, y):
        return np.median(np.abs(y - y_hat))

    def r2(y_hat, y):
        return 1 - mse(y_hat, y) / np.var(y)

    def all_score(y_hat, y):
        return [
            mse(y_hat, y), explained_variance_score(y_hat, y),
            max_error(y_hat, y), mae(y_hat, y), median_ae(y_hat, y), r2(y_hat, y)
        ]

    noise = 0.3
    num = 30
    k = 9

    lbds = [0., 1e-3, 1e-2, 1e-1, 2e-1, 3e-1]

    x_train, y_train = data.get_data_set(
        num, miu=0., sigma=noise, outlier=0., shuffle=True, outer_miu=1., outer_sigma=1.
    )
    plt.scatter(x_train, y_train, s=2., c=[[0.6, 0.2, 0.2] for _ in range(x_train.shape[0])])

    x_test, y_test = data.get_data_set(
        num, miu=0., sigma=noise, outlier=0., shuffle=True, outer_miu=1., outer_sigma=1.
    )
    plt.scatter(x_test, y_test, s=2., c=[[0.1, 0.6, 0.2] for _ in range(x_train.shape[0])])

    model = poly_regression.Poly()

    for lbd in lbds:
        model.ridge_regression(x_train, y_train, lbd, k=k)   # increase k

        print(all_score(model(x_train), y_train))

        plt.plot(np.linspace(-4.3, 4.3, 100), model(np.linspace(-4.3, 4.3, 100)), linewidth=0.5)

        y_hat = model(x_test)

        print(all_score(y_hat, y_test))

        print("")

    plt.legend(['train_set', 'test_set'] + ["lbd={}".format(lbd) for lbd in lbds])
    plt.show()

# over_fit_lbd()

def over_fit_lasso():

    def mse(y_hat, y):
        return np.mean((y_hat - y) ** 2)

    def explained_variance_score(y_hat, y):
        return 1 - np.var(y - y_hat) / np.var(y)

    def max_error(y_hat, y):
        return np.max(np.abs(y - y_hat))

    def mae(y_hat, y):
        return np.mean(np.abs(y - y_hat))

    def median_ae(y_hat, y):
        return np.median(np.abs(y - y_hat))

    def r2(y_hat, y):
        return 1 - mse(y_hat, y) / np.var(y)

    def all_score(y_hat, y):
        return [
            mse(y_hat, y), explained_variance_score(y_hat, y),
            max_error(y_hat, y), mae(y_hat, y), median_ae(y_hat, y), r2(y_hat, y)
        ]

    noise = 0.5
    num = 80
    k = 5

    lbds = [0., 1e-2, 1e-1, 2e-1, 5e-1, 1., 1e1, 1e2]

    x_train, y_train = data.get_data_set(
        num, miu=0., sigma=noise, outlier=0., shuffle=True, outer_miu=1., outer_sigma=1.
    )
    plt.scatter(x_train, y_train, s=2., c=[[0.6, 0.2, 0.2] for _ in range(x_train.shape[0])])

    x_test, y_test = data.get_data_set(
        num, miu=0., sigma=noise, outlier=0., shuffle=True, outer_miu=1., outer_sigma=1.
    )
    plt.scatter(x_test, y_test, s=2., c=[[0.1, 0.6, 0.2] for _ in range(x_train.shape[0])])

    model = poly_regression.Poly()

    for lbd in lbds:
        model.lasso(x_train, y_train, lbd, k=k, max_iter=20000, lr=1e-5)   # increase k

        print(all_score(model(x_train), y_train))

        plt.plot(np.linspace(-4.3, 4.3, 100), model(np.linspace(-4.3, 4.3, 100)), linewidth=0.5)

        y_hat = model(x_test)

        print(all_score(y_hat, y_test))

        print("")

    plt.legend(['train_set', 'test_set'] + ["lbd={}".format(lbd) for lbd in lbds])
    plt.show()


# over_fit_lasso()


def over_fit_data():

    def mse(y_hat, y):
        return np.mean((y_hat - y) ** 2)

    def explained_variance_score(y_hat, y):
        return 1 - np.var(y - y_hat) / np.var(y)

    def max_error(y_hat, y):
        return np.max(np.abs(y - y_hat))

    def mae(y_hat, y):
        return np.mean(np.abs(y - y_hat))

    def median_ae(y_hat, y):
        return np.median(np.abs(y - y_hat))

    def r2(y_hat, y):
        return 1 - mse(y_hat, y) / np.var(y)

    def all_score(y_hat, y):
        return [
            mse(y_hat, y), explained_variance_score(y_hat, y),
            max_error(y_hat, y), mae(y_hat, y), median_ae(y_hat, y), r2(y_hat, y)
        ]

    noise = 0.3
    nums = [5, 10, 20, 50, 100, 500]
    k = 4

    lbd = 0


    # plt.scatter(x_train, y_train, s=2., c=[[0.6, 0.2, 0.2] for _ in range(x_train.shape[0])])

    x_test, y_test = data.get_data_set(
        100, miu=0., sigma=noise, outlier=0., shuffle=True, outer_miu=1., outer_sigma=1.
    )
    plt.scatter(x_test, y_test, s=2., c=[[0.1, 0.6, 0.2] for _ in range(x_test.shape[0])])

    model = poly_regression.Poly()

    for num in nums:
        x_train, y_train = data.get_data_set(
            num, miu=0., sigma=noise, outlier=0., shuffle=True, outer_miu=1., outer_sigma=1.
        )

        model.ridge_regression(x_train, y_train, lbd, k=k)   # increase k

        print(all_score(model(x_train), y_train))

        plt.plot(np.linspace(-4.3, 4.3, 100), model(np.linspace(-4.3, 4.3, 100)), linewidth=0.5)

        y_hat = model(x_test)

        print(all_score(y_hat, y_test))

        print("")

    plt.legend(['test_set'] + ["train_size={}".format(s) for s in nums])
    plt.show()


# over_fit_data()


def ridge_w():

    noise = 0.1
    num = 100
    lbds = [0., 1e-3, 1e-2, 1e-1, 2e-1, 3e-1, 4e-1, 5e-1, 8e-1, 1., 2, 3]

    x, y = data.get_data_set(
        num, miu=0., sigma=noise, outlier=0., shuffle=True, outer_miu=1., outer_sigma=1.
    )

    # plt.scatter(x, y, s=2., c=[[0.6, 0.2, 0.2] for _ in range(x.shape[0])])

    model = poly_regression.Poly()

    ws = []

    for lbd in lbds:
        model.ridge_regression(x, y, lbd=lbd, k=4, is_iter=False)

        ws.append(model.w.reshape([-1]))

    ws = np.array(ws).T

    for l in ws:
        plt.plot(lbds, l, linewidth=1.)

    plt.legend(['w{}'.format(i) for i in range(ws.shape[0])])
    plt.show()


# ridge_w()