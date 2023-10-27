import numpy as np
import matplotlib.pyplot as plt


# imgs = np.load(r"./gen.npy")
imgs = np.load(r"./dif.npy")

idxs = [0, 14, 2, 3, 45, 22, 5, 25, 26, 34, 28, 11, 16, 13, 34, 32]
idxs = range(30, 30+16)
plt.subplots_adjust(wspace=0.1, hspace=0.05)
plt.figure(figsize=(16, 16))

for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.axis("off")
    plt.imshow(imgs[idxs[i]].reshape([28, 28]))

plt.show()

imgs = np.load(r"./dif.npy")

idxs = range(64)
plt.subplots_adjust(wspace=0.1, hspace=0.05)
plt.figure(figsize=(16, 16))

for i in range(64):
    plt.subplot(8, 8, i+1)
    plt.axis("off")
    plt.imshow(imgs[idxs[i]].reshape([28, 28]))

plt.show()




