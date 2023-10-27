import mlp
import data
import numpy as np
import matplotlib.pyplot as plt


n = 300
h = 10
test_rate = 0.2

x = np.random.random([n, h])

y = (x.reshape([n, 1, h]) @ np.random.randn(h, h) @ x.reshape([n, h, 1])).reshape([n, 1])

y = y + np.random.randn(n, 1) * 0.01

x_train = x[:int(n * (1 - test_rate)), :]
y_train = y[:int(n * (1 - test_rate)), :]
x_test = x[int(n * (1 - test_rate)):, :]
y_test = y[int(n * (1 - test_rate)):, :]

plt.figure(figsize=(12, 6))


def train(model):
    # model = mlp.MLP(h, 1, [16], dropout=0.1, act=mlp.Swish, norm=True)

    loss_func = mlp.MSE()
    # loss_func = mlp.L1Loss()

    opt = mlp.Adam(model.layers, lr=1e-3, l2=0.)

    losses = []


    for epoch in range(1000):
        y_hat = model.forward(x_train)

        loss = loss_func(y_hat, y_train)

        dy = loss_func.backward()

        dy = model.backward(dy)

        opt.step()

        print(loss)
        losses.append(loss)

    plt.plot(losses, linewidth=0.5)

    model.eval()

    y_hat = model.forward(x_test, grad=False)

    loss = loss_func(y_hat, y_test)

    print(loss)


train(mlp.MLP(h, 1, [16], dropout=0.1, act=mlp.Swish, norm=True))
train(mlp.MLP(h, 1, [16], dropout=0.1, act=mlp.LeakyRelu, norm=True))
train(mlp.MLP(h, 1, [16], dropout=0.1, act=mlp.Relu, norm=True))
train(mlp.MLP(h, 1, [16], dropout=0.1, act=mlp.Sigmoid, norm=True))
train(mlp.MLP(h, 1, [16], dropout=0.1, act=mlp.Tanh, norm=True))

plt.legend(['Swish', 'LeakyRelu', 'Relu', 'Sigmoid', 'Tanh'])


plt.show()






