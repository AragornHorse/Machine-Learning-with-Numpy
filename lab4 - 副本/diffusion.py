import numpy as np
import data
import mlp

batch_size = 512

T = 200

d_model = 64

betas = np.linspace(0.001, 0.02, T)
alphas = 1 - betas
alphas_mult = np.copy(alphas)
for i in range(1, alphas_mult.shape[0]):
    alphas_mult[i] *= alphas_mult[i-1]

# print(betas)
# print(alphas)
# print(alphas_mult)


def get_t_emb(t, d_model):
    i = np.array(range(d_model))
    te = np.zeros([t.shape[0], 2 * d_model])
    te[:, :d_model] = np.sin(t.reshape([-1, 1]) / (1e4 ** (i.reshape([1, -1]) / 2 / d_model)))
    te[:, d_model:] = np.cos(t.reshape([-1, 1]) / (1e4 ** (i.reshape([1, -1]) / 2 / d_model)))
    return te

# print(get_t_emb(np.array([1, 2, 3, 4]), 5))


def pollute(x, t=None):
    if isinstance(t, int):
        t = np.full([x.shape[0]], t)
    if t is None:
        t = np.random.randint(0, T, [x.shape[0]])

    noise = np.random.randn(*(x.shape))   # n, h
    t_ = t.reshape([-1]).astype(int)
    alpha = alphas_mult[t_].reshape([-1, 1])
    x_ = np.sqrt(alpha) * x + np.sqrt(1 - alpha) * noise
    return x_, noise, t


def generate(x, model):
    model.eval()
    for t in reversed(range(T)):
        # print(np.concatenate([x, np.full([x.shape[0], 1], t) / T], axis=1).shape)
        z = model(np.concatenate([x, get_t_emb(np.full([x.shape[0], 1], t), d_model)], axis=1))

        if t > 1:
            x = 1 / np.sqrt(alphas[t]) * (x - (1 - alphas[t]) / np.sqrt(1 - alphas_mult[t]) * z) +\
                np.sqrt((1 - alphas[t]) * (1 - alphas_mult[t-1]) / (1 - alphas_mult[t])) * np.random.randn(*(x.shape))
        else:
            x = 1 / np.sqrt(alphas[t]) * (x - (1 - alphas[t]) / np.sqrt(1 - alphas_mult[t]) * z)

    return x


loader = data.DataLoader(data.train_x / 255, data.train_y, batch_size)

# loss_func = mlp.MSE()
loss_func = mlp.L1Loss()

model = mlp.MLP(28 ** 2 + d_model * 2, 28 ** 2, [1024], dropout=0., act=mlp.Relu, norm=True, end_norm=False)
# model = mlp.Sequence(
#     mlp.ResidualBlock(28 ** 2 + d_model * 2, 1024, [512, 512], dropout=0., act=mlp.Swish, norm=True),
#     # mlp.BatchNorm(1024),
#     mlp.ResidualBlock(1024, 28 ** 2, [512, 512], dropout=0., act=mlp.Swish, norm=True)
# )
# model = mlp.MLP(28 ** 2 + d_model * 2, 28 ** 2, [1024, 1024], dropout=0., act=mlp.Relu, norm=True, end_norm=True)
# model = mlp.MLP(28 ** 2 + d_model * 2, 28 ** 2, [1024, 1024], dropout=0., act=mlp.Relu, norm=True, end_norm=True)

# model.save(r"./para")
# model.load(r"./para")

print(model)

opt = mlp.Adam(model.layers, lr=1e-3)

lrs = np.linspace(1e-3, 1e-5, 1000)
lrs = np.concatenate([np.full([40, 1], 1e-3), np.linspace(1e-3, 1e-4, 30).reshape([30, 1]), np.linspace(1e-4, 3e-5, 30).reshape([30, 1]), np.linspace(3e-5, 1e-6, 1000).reshape([-1, 1])], 0).reshape([-1])

# lrs = np.concatenate([np.full([5, 1], 1e-6).reshape([-1, 1]), np.linspace(1e-6, 1e-5, 30).reshape([-1, 1]), np.linspace(1e-5, 3e-6, 100).reshape([-1, 1]), np.linspace(3e-6, 2e-6, 100).reshape([-1, 1]), np.linspace(2e-6, 1e-7, 100).reshape([-1, 1])], 0).reshape([-1])
# lrs *= 1e-1


# print(16 * 1024)

for epoch in range(10):

    opt.lr = lrs[epoch]

    for i in range(10):
        data = loader[i]

        x, _ = data
        x, z, t = pollute(x)

        t = get_t_emb(t, d_model)
        x = np.concatenate([x, t], axis=1)

        # print(x.shape)

        z_pred = model(x)

        loss = loss_func(z_pred, z)

        dy = loss_func.backward()

        dx = model.backward(dy)

        opt.step()

        print("epoch{}: {}".format(epoch, loss / 28 ** 2))

        # print(z)
        # print(z_pred)
        # print('')

import matplotlib.pyplot as plt

x, _ = loader[0]

# print(x[0])

# x, _, _ = pollute(x, T-1)
x = np.random.randn(64, 28 ** 2)

# print(x[0])

plt.imshow(x[0, :].reshape([28, 28]))
plt.show()

x = generate(x[:, :], model)

np.save(r"./dif.npy", x)

plt.imshow(x[0].reshape([28, 28]) * 255)
plt.show()

imgs = np.load(r"./dif.npy")

idxs = [24, 1, 2, 3, 4, 22, 23, 25, 26, 27, 28, 11, 12, 13, 14, 15]
plt.subplots_adjust(wspace=0.1, hspace=0.05)
plt.figure(figsize=(16, 16))

for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.axis("off")
    plt.imshow(imgs[idxs[i]].reshape([28, 28]))

plt.show()

input("save >>> ")
model.save(r"./para")






