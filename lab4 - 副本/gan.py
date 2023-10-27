import mlp
import data
import numpy as np


batch_size = 128
fake_size = 64

np.random.seed(2)

loader = data.DataLoader(data.train_x, data.train_y, batch_size=batch_size)

generator = mlp.MLP(28 ** 2, 28 ** 2, [256], dropout=0., act=mlp.Swish)
discriminator = mlp.MLP(28 ** 2, 2, [8], dropout=0., act=mlp.Swish)

loss_func = mlp.CrossEntropy()

gen_opt = mlp.Adam(generator.layers, lr=1e-4, l2=0.)
dis_opt = mlp.Adam(discriminator.layers, lr=1e-4, l2=0.)

noise = np.random.random([fake_size, 28 ** 2]) / 100
for i in range(fake_size):
    noise[i] = (i % 10) / 10 + np.random.random([28 ** 2]) / 1000

dis_label = np.concatenate([np.ones([batch_size, 1]), np.zeros([fake_size, 1])], 0).reshape([-1]).astype("int")
gen_label = np.ones([fake_size, 1]).astype("int")


x, y = loader[0]


for epoch in range(50):

    fake_img = generator.forward(noise)

    for i in range(10):

        # x, y = loader[i]

        x_ = np.concatenate([x, fake_img], axis=0)

        y_dis = discriminator.forward(x_)

        loss = loss_func(y_dis, dis_label)

        if loss < 0.2:
            break

        dy = loss_func.backward()

        discriminator.backward(dy)

        dis_opt.step()

        print("dis_loss:{}".format(loss))

    # gen_opt.lr = 1e-3 / (epoch + 1)

    for i in range(60):

        fake_img = generator.forward(noise)

        y_dis = discriminator.forward(fake_img)

        loss = loss_func(y_dis, gen_label)

        if loss < 0.5:
            break

        dy = loss_func.backward()

        dy = discriminator.backward(dy)

        generator.backward(dy)

        gen_opt.step()

        print("gen_loss:{}".format(loss))

    if epoch % 8 == 1:

        import matplotlib.pyplot as plt

        plt.imshow(fake_img[0].reshape([28, 28]))

        plt.show()

plt.imshow(fake_img[0].reshape([28, 28]))

plt.show()

np.save(r"./gen.npy", fake_img)









