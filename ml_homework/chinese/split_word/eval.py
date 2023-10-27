import jieba
import numpy as np

rst = []
ori = []
with open(r"./rst.txt", 'r', encoding='utf-8') as f:
    i = 0
    for line in f:
        i += 1
        if i % 3 == 0:
            continue
        if i % 3 == 1:
            ori.append(line)
        else:
            rst.append(line.split(','))


def aou(lst1, lst2):
    u = 0
    n = 0
    for i in lst1:
        if i in lst2:
            n += 1
            u += 1
        else:
            u += 1

    for i in lst2:
        if i in lst1:
            n += 1
            u += 1
        else:
            u += 1
    return n / u


def I(lst1, lst2):
    dic1 = []
    for i in lst1:
        if i not in dic1:
            dic1.append(i)

    dic2 = []
    for i in lst2:
        if i not in dic2:
            dic2.append(i)

    dic1 = {k:v for v, k in enumerate(dic1)}
    dic2 = {k: v for v, k in enumerate(dic2)}

    n = np.zeros([len(dic1), len(dic2)])

    for w_ in lst1:
        for w in w_:
            for w__ in lst1:
                if w in w__:
                    i = dic1[w__]
                    break
            for w__ in lst2:
                if w in w__:
                    j = dic2[w__]
                    break
            n[i, j] += 1

    P = n / np.sum(n)

    p_x = np.sum(P, axis=0)
    p_y = np.sum(P, axis=1)

    H_X = -np.sum(p_x * np.log(p_x + 1e-30))
    H_y = -np.sum(p_y * np.log(p_y + 1e-30))
    H_xy = -np.sum(P * np.log(P + 1e-30))

    return H_X + H_y - H_xy


aous = []
Is = []

for i in range(len(rst)):
    o = ori[i]
    y_hat = rst[i]
    y = list(jieba.cut(o))

    print(o)
    print(y)
    print(y_hat)

    print(aou(y, y_hat))
    print(I(y, y_hat))
    print("")

    aous.append(aou(y, y_hat))
    Is.append(I(y, y_hat))


print(np.mean(np.array(aous)), np.mean(np.array(Is)))




# print(np.__version__)








