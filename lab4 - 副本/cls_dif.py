import mlp
import numpy as np
import data
import matplotlib.pyplot as plt

loader = data.DataLoader(data.train_x, data.train_y, batch_size=256)

model = mlp.MLP(28 ** 2, 10, [128, 64], dropout=0.2, act=mlp.Swish)

loss_func = mlp.CrossEntropy()

# opt = mlp.GradientDescent(model.layers, lr=1e-1, l2=0.)
opt = mlp.Adam(model.layers, lr=1e-2, l2=0.)
# opt = mlp.Lion(model.layers, lr=1e-2, l2=0.)
# opt = mlp.AdaGrad(model.layers, lr=1e-1, l2=0.)
# opt = mlp.RMSProp(model.layers, lr=1e-2, l2=0.)

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
    if i == 400:
        break

import data

print("eval")
model.eval()

y_hat = model.forward(data.test_x[:256], False)
y_hat = mlp.Softmax().forward(y_hat[:256], False)

loss = loss_func(y_hat, data.test_y[:256], False)

acc = np.mean((np.argmax(y_hat, axis=-1) == data.test_y[:256]).astype(float))

print(loss, acc)


imgs = np.load(r"./dif.npy") * 255

y_hat = model.forward(((imgs > 200) * 255), False)
cls = np.argmax(y_hat, axis=1)

idxs = range(64)
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.figure(figsize=(16, 18))

for i in range(64):
    plt.subplot(8, 8, i+1)
    plt.title("{}".format(cls[i]))
    plt.axis("off")
    plt.imshow(imgs[idxs[i]].reshape([28, 28]))

plt.show()



