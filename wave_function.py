# -*- encoding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# a0 = 5.29
a0 = 5.29


def wave100(x, z):
    y = np.exp(-(z * 2 * x) / a0) * np.sqrt(z ** 3 / (np.pi * (a0 ** 3)))
    return y


def wave200(x, z):
    y = np.exp(-(z * x) / a0) * np.sqrt(
        z ** 3 / (32 * np.pi * (a0 ** 3))) * (2 - z * 2 * x / a0)
    return y


def wave(x, z):
    y = wave100(x, z)
    if 3 <= z:
        y += wave200(x, z)
    return 4 * np.pi * ((2 * x) ** 2) * (y ** 2)


def main():
    x = np.arange(0, 5, 0.1)
    # wave100(z=1)
    # x, y = wave100(z=6)
    # plt.plot(x,y)
    # plt.show()
    # wave100(z=7)
    # wave100(z=8)
    y1 = wave(x, z=6)
    print(y1)
    # plt.plot(x, y1)
    # plt.show()
    y2 = wave(4.9 - x, z=8)
    print(y2)
    print(y1 + y2)
    # plt.plot(x, y2)
    # plt.show()
    # plt.plot(x, y1 + y2)
    # plt.show()


if __name__ == '__main__':
    main()
