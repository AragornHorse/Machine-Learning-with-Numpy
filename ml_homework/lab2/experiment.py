import logistic
import data
import matplotlib.pyplot as plt
import numpy as np


def learning_curve():

    k = 20

    lbd = 0.

    it_num = 100

    pca = logistic.PCA()

    pca.get_w(data.train_x, k=k)

    loader = data.DataLoader(pca(data.train_x, k=k, update_w=False), data.train_y, batch_size=128)

    losses = []
    accs = []
    model = logistic.Logistic(in_size=k, out_size=10)
    for i in range(it_num):
        x, y = loader[i]
        loss, acc = model.grad_train(x, y, weight_decay=lbd, lr=1e-5)
        print(loss, acc)
        losses.append(loss)
        accs.append(acc)

    plt.plot(losses, linewidth=1)
    plt.plot(accs, linewidth=1, linestyle='--')

    losses = []
    accs = []
    model = logistic.Logistic(in_size=k, out_size=10)
    for i in range(it_num):
        x, y = loader[i]

        loss, acc = model.conjugate_gradient(x, y, weight_decay=lbd, lr=5e-7)
        print(loss, acc)
        losses.append(loss)
        accs.append(acc)

    plt.plot(losses, linewidth=1)
    plt.plot(accs, linewidth=1, linestyle='--')

    losses = []
    accs = []
    model = logistic.Logistic(in_size=k, out_size=10)
    for i in range(it_num):
        x, y = loader[i]

        loss, acc = model.train_newton(x, y, weight_decay=lbd)
        print(loss, acc)
        losses.append(loss)
        accs.append(acc)

    plt.plot(losses, linewidth=1)
    plt.plot(accs, linewidth=1, linestyle='--')

    plt.legend(['GD loss', 'GD acc', 'CG loss', 'CG acc', 'NT loss', 'NT acc'])
    plt.show()


# learning_curve()


def show_w():
    loader = data.DataLoader(data.train_x, data.train_y, batch_size=256)

    losses = []
    accs = []

    model = logistic.Logistic(in_size=28**2, out_size=10)
    for i in range(200):
        x, y = loader[i]
        loss, acc = model.grad_train(x, y, weight_decay=1e2, lr=1e-5)
        print(loss, acc)
        losses.append(loss)
        accs.append(acc)

    plt.figure(figsize=(10, 4))
    plt.subplots_adjust(wspace=0.1, hspace=0.05)
    ws = model.w.T
    for i, w in enumerate(ws):
        plt.axis("off")
        plt.subplot(2, 5, i+1)
        plt.axis("off")
        plt.imshow(w[:28**2].reshape([28, 28]))

    plt.show()

# show_w()


def eval():

    def all(p_hat, y):
        y_hat = np.argmax(p_hat, 1)

        acc = np.mean(y_hat == y)

        y_onehot = np.zeros_like(p_hat)
        for i, j in enumerate(y):
            y_onehot[i, j] = 1

        loss = - np.mean(y_onehot * np.log(p_hat + 1e-30)) * p_hat.shape[1]

        return loss, acc

    loader = data.DataLoader(data.train_x, data.train_y, batch_size=256)

    losses = []
    accs = []

    model = logistic.Logistic(in_size=28**2, out_size=10)
    for i in range(1000):
        x, y = loader[i]
        loss, acc = model.grad_train(x, y, weight_decay=100, lr=1e-5)
        print(loss, acc)
        losses.append(loss)
        accs.append(acc)

    p_hat = model(data.train_x)

    print(all(p_hat, data.train_y))


    p_hat = model(data.test_x)

    print(all(p_hat, data.test_y))


eval()

# print(np.__version__)
