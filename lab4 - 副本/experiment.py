import numpy as np
import data
import mlp
import matplotlib.pyplot as plt


def lc_act():
    plt.figure(figsize=(12, 6))

    def train(model):
        import data
        loader = data.DataLoader(data.train_x, data.train_y, batch_size=256)

        loss_func = mlp.CrossEntropy()

        # opt = mlp.GradientDescent(model.layers, lr=1e-1, l2=0.)
        opt = mlp.Adam(model.layers, lr=1e-3, l2=0.)
        # opt = mlp.Lion(model.layers, lr=1e-3, l2=0.)

        i = 0

        losses = []
        accs = []

        for data in loader:
            x, y = data
            y_hat = model.forward(x)

            loss = loss_func.forward(y_hat, y)

            acc = np.mean(np.argmax(y_hat, axis=-1) == y)

            losses.append(loss)
            accs.append(acc)

            dy = loss_func.backward()

            model.backward(dy)

            opt.step()

            print("loss:{}, acc:{}".format(loss, acc))
            i += 1
            if i == 200:
                break

        plt.plot(losses, linewidth=0.5)
        plt.plot(accs, linewidth=0.5, linestyle='--')

    train(mlp.MLP(28 ** 2, 10, [64, 32], dropout=0.1, act=mlp.Swish))
    train(mlp.MLP(28 ** 2, 10, [64, 32], dropout=0.1, act=mlp.LeakyRelu))
    train(mlp.MLP(28 ** 2, 10, [64, 32], dropout=0.1, act=mlp.Relu))
    train(mlp.MLP(28 ** 2, 10, [64, 32], dropout=0.1, act=mlp.Sigmoid))
    train(mlp.MLP(28 ** 2, 10, [64, 32], dropout=0.1, act=mlp.Tanh))

    plt.legend([
        'Swish loss', 'Swish acc', 'LeakyRelu loss', 'LeakyRelu acc', 'Relu loss', 'Relu acc',
        'Sigmoid loss', 'Sigmoid acc', 'Tanh loss', 'Tanh acc'
    ])

    plt.show()

# lc_act()


def lc_norm():
    plt.figure(figsize=(12, 6))

    def train(model, lr=1e-4):
        import data
        loader = data.DataLoader(data.train_x, data.train_y, batch_size=256)

        loss_func = mlp.CrossEntropy()

        # opt = mlp.GradientDescent(model.layers, lr=1e-1, l2=0.)
        opt = mlp.Adam(model.layers, lr=lr, l2=0.)
        # opt = mlp.Lion(model.layers, lr=1e-3, l2=0.)

        i = 0

        losses = []
        accs = []

        for data in loader:
            x, y = data
            y_hat = model.forward(x)

            loss = loss_func.forward(y_hat, y)

            acc = np.mean(np.argmax(y_hat, axis=-1) == y)

            losses.append(loss)
            accs.append(acc)

            dy = loss_func.backward()

            model.backward(dy)

            opt.step()

            print("loss:{}, acc:{}".format(loss, acc))
            i += 1
            if i == 200:
                break

        plt.plot(losses, linewidth=0.5)
        plt.plot(accs, linewidth=0.5, linestyle='--')

    # train(mlp.MLP(28 ** 2, 10, [64, 32], dropout=0.1, act=mlp.Swish))
    # train(mlp.MLP(28 ** 2, 10, [64, 32], dropout=0.1, act=mlp.Swish, norm=False))
    # # train(mlp.MLP(28 ** 2, 10, [64, 32], dropout=0.1, act=mlp.LeakyRelu))
    # train(mlp.MLP(28 ** 2, 10, [64, 32], dropout=0.1, act=mlp.Relu))
    # train(mlp.MLP(28 ** 2, 10, [64, 32], dropout=0.1, act=mlp.Relu, norm=False))

    train(mlp.MLP(28 ** 2, 10, [64, 32], dropout=0.1, act=mlp.Sigmoid))
    train(mlp.MLP(28 ** 2, 10, [64, 32], dropout=0.1, act=mlp.Sigmoid, norm=False))

    train(mlp.MLP(28 ** 2, 10, [64, 32], dropout=0.1, act=mlp.Tanh))
    train(mlp.MLP(28 ** 2, 10, [64, 32], dropout=0.1, act=mlp.Tanh, norm=False))

    plt.legend([
        # 'Swish loss', 'Swish acc', 'Swish loss (no norm)', 'Swish acc (no norm)', 'Relu loss', 'Relu acc',
        # 'Relu loss (no norm)', 'Relu acc (no norm)',
        'Sigmoid loss', 'Sigmoid acc',
        'Sigmoid loss (no norm)', 'Sigmoid acc (no norm)',
        'Tanh loss', 'Tanh acc', 'Tanh loss (no norm)', 'Tanh acc (no norm)'
    ])

    plt.show()

# lc_norm()

def lc_layer():
    plt.figure(figsize=(12, 6))

    def train(model, lr=1e-4):
        import data
        loader = data.DataLoader(data.train_x, data.train_y, batch_size=256)

        loss_func = mlp.CrossEntropy()

        # opt = mlp.GradientDescent(model.layers, lr=1e-1, l2=0.)
        opt = mlp.Adam(model.layers, lr=lr, l2=0.)
        # opt = mlp.Lion(model.layers, lr=1e-3, l2=0.)

        i = 0

        losses = []
        accs = []

        for data in loader:
            x, y = data
            y_hat = model.forward(x)

            loss = loss_func.forward(y_hat, y)

            acc = np.mean(np.argmax(y_hat, axis=-1) == y)

            losses.append(loss)
            accs.append(acc)

            dy = loss_func.backward()

            model.backward(dy)

            opt.step()

            print("loss:{}, acc:{}".format(loss, acc))
            i += 1
            if i == 200:
                break

        plt.plot(losses, linewidth=0.5)
        plt.plot(accs, linewidth=0.5, linestyle='--')

    # train(mlp.MLP(28 ** 2, 10, [64, 32], dropout=0.1, act=mlp.Swish))
    # train(mlp.MLP(28 ** 2, 10, [64, 32], dropout=0.1, act=mlp.Swish, norm=False))
    # # train(mlp.MLP(28 ** 2, 10, [64, 32], dropout=0.1, act=mlp.LeakyRelu))
    train(mlp.MLP(28 ** 2, 10, [64, 32], dropout=0.1, act=mlp.LeakyRelu))
    train(mlp.MLP(28 ** 2, 10, [64, 32], dropout=0.1, act=mlp.LeakyRelu, norm=False))

    train(mlp.MLP(28 ** 2, 10, [64], dropout=0.1, act=mlp.LeakyRelu))
    train(mlp.MLP(28 ** 2, 10, [64], dropout=0.1, act=mlp.LeakyRelu, norm=False))

    train(mlp.MLP(28 ** 2, 10, [64, 32], dropout=0.1, act=mlp.Sigmoid))
    train(mlp.MLP(28 ** 2, 10, [64, 32], dropout=0.1, act=mlp.Sigmoid, norm=False))

    train(mlp.MLP(28 ** 2, 10, [64], dropout=0.1, act=mlp.Sigmoid))
    train(mlp.MLP(28 ** 2, 10, [64], dropout=0.1, act=mlp.Sigmoid, norm=False))

    # train(mlp.MLP(28 ** 2, 10, [64, 32], dropout=0.1, act=mlp.Tanh))
    # train(mlp.MLP(28 ** 2, 10, [64, 32], dropout=0.1, act=mlp.Tanh, norm=False))

    plt.legend([
        # 'Swish loss', 'Swish acc', 'Swish loss (no norm)', 'Swish acc (no norm)',
        'Relu loss 4-layer', 'Relu acc 4-layer',
        'Relu loss (no norm) 4-layer', 'Relu acc (no norm) 4-layer',
        'Relu loss 3-layer', 'Relu acc 3-layer',
        'Relu loss (no norm) 3-layer', 'Relu acc (no norm) 3-layer',
        'Sigmoid loss 4-layer', 'Sigmoid acc 4-layer',
        'Sigmoid loss (no norm) 4-layer', 'Sigmoid acc (no norm) 4-layer',
        'Sigmoid loss 3-layer', 'Sigmoid acc 3-layer',
        'Sigmoid loss (no norm) 3-layer', 'Sigmoid acc (no norm) 3-layer'
        # 'Tanh loss', 'Tanh acc', 'Tanh loss (no norm)', 'Tanh acc (no norm)'
    ])

    plt.show()

# lc_layer()

def lc_opt():
    plt.figure(figsize=(12, 6))

    def train(model, opt, lr=1e-3):
        import data
        loader = data.DataLoader(data.train_x, data.train_y, batch_size=256)

        loss_func = mlp.CrossEntropy()

        # opt = mlp.GradientDescent(model.layers, lr=1e-1, l2=0.)
        opt = opt(model.layers, lr=lr, l2=0.)
        # opt = mlp.Lion(model.layers, lr=1e-3, l2=0.)

        i = 0

        losses = []
        accs = []

        for data in loader:
            x, y = data
            y_hat = model.forward(x)

            loss = loss_func.forward(y_hat, y)

            acc = np.mean(np.argmax(y_hat, axis=-1) == y)

            losses.append(loss)
            accs.append(acc)

            dy = loss_func.backward()

            model.backward(dy)

            opt.step()

            print("loss:{}, acc:{}".format(loss, acc))
            i += 1
            if i == 200:
                break

        plt.plot(losses, linewidth=0.5)
        plt.plot(accs, linewidth=0.5, linestyle='--')

    # train(mlp.MLP(28 ** 2, 10, [64, 32], dropout=0.1, act=mlp.Swish))
    # train(mlp.MLP(28 ** 2, 10, [64, 32], dropout=0.1, act=mlp.Swish, norm=False))
    # # train(mlp.MLP(28 ** 2, 10, [64, 32], dropout=0.1, act=mlp.LeakyRelu))
    train(mlp.MLP(28 ** 2, 10, [64, 32], dropout=0.1, act=mlp.LeakyRelu), mlp.Adam)

    train(mlp.MLP(28 ** 2, 10, [64, 32], dropout=0.1, act=mlp.Sigmoid), mlp.Adam)

    train(mlp.MLP(28 ** 2, 10, [64, 32], dropout=0.1, act=mlp.LeakyRelu), mlp.GradientDescent)

    train(mlp.MLP(28 ** 2, 10, [64, 32], dropout=0.1, act=mlp.Sigmoid), mlp.GradientDescent)

    train(mlp.MLP(28 ** 2, 10, [64, 32], dropout=0.1, act=mlp.LeakyRelu), mlp.Lion)

    train(mlp.MLP(28 ** 2, 10, [64, 32], dropout=0.1, act=mlp.Sigmoid), mlp.Lion)

    # train(mlp.MLP(28 ** 2, 10, [64, 32], dropout=0.1, act=mlp.Tanh))
    # train(mlp.MLP(28 ** 2, 10, [64, 32], dropout=0.1, act=mlp.Tanh, norm=False))

    plt.legend([
        'Relu loss Adam', 'Relu acc Adam',
        'Sigmoid loss Adam', 'Sigmoid acc Adam',
        'Relu loss GD', 'Relu acc GD',
        'Sigmoid loss GD', 'Sigmoid acc GD',
        'Relu loss Lion', 'Relu acc Lion',
        'Sigmoid loss Lion', 'Sigmoid acc Lion',
    ])

    plt.show()

# lc_opt()


def lc_all_opt():
    plt.figure(figsize=(12, 6))

    def train(model, opt, lr=1e-3):
        import data
        loader = data.DataLoader(data.train_x, data.train_y, batch_size=256)

        loss_func = mlp.CrossEntropy()

        # opt = mlp.GradientDescent(model.layers, lr=1e-1, l2=0.)
        opt = opt(model.layers, lr=lr, l2=0.)
        # opt = mlp.Lion(model.layers, lr=1e-3, l2=0.)

        i = 0

        losses = []
        accs = []

        for data in loader:
            x, y = data
            y_hat = model.forward(x)

            loss = loss_func.forward(y_hat, y)

            acc = np.mean(np.argmax(y_hat, axis=-1) == y)

            losses.append(loss)
            accs.append(acc)

            dy = loss_func.backward()

            model.backward(dy)

            opt.step()

            print("loss:{}, acc:{}".format(loss, acc))
            i += 1
            if i == 200:
                break

        plt.plot(losses, linewidth=0.5)
        plt.plot(accs, linewidth=0.5, linestyle='--')

    # train(mlp.MLP(28 ** 2, 10, [64, 32], dropout=0.1, act=mlp.Swish))
    # train(mlp.MLP(28 ** 2, 10, [64, 32], dropout=0.1, act=mlp.Swish, norm=False))
    # # train(mlp.MLP(28 ** 2, 10, [64, 32], dropout=0.1, act=mlp.LeakyRelu))
    train(mlp.MLP(28 ** 2, 10, [64, 32], dropout=0.1, act=mlp.LeakyRelu), mlp.Adam)

    train(mlp.MLP(28 ** 2, 10, [64, 32], dropout=0.1, act=mlp.LeakyRelu), mlp.GradientDescent)

    train(mlp.MLP(28 ** 2, 10, [64, 32], dropout=0.1, act=mlp.LeakyRelu), mlp.Lion)

    train(mlp.MLP(28 ** 2, 10, [64, 32], dropout=0.1, act=mlp.LeakyRelu), mlp.AdaGrad)

    train(mlp.MLP(28 ** 2, 10, [64, 32], dropout=0.1, act=mlp.LeakyRelu), mlp.RMSProp)


    # train(mlp.MLP(28 ** 2, 10, [64, 32], dropout=0.1, act=mlp.Tanh))
    # train(mlp.MLP(28 ** 2, 10, [64, 32], dropout=0.1, act=mlp.Tanh, norm=False))

    plt.legend([
        'loss Adam', 'acc Adam',
        'loss GD', 'acc GD',
        'loss Lion', 'acc Lion',
        'loss AdaGrad', 'acc AdaGrad',
        'loss RMSProp', 'acc RMSProp',
    ])

    plt.show()

lc_all_opt()


