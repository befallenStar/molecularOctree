# -*- encoding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# a0 = 5.29
a0 = 5.29


def wave100(z, x):
    y = np.exp(-(z * 4 * x) / a0) * np.sqrt(z ** 3 / (np.pi * (a0 ** 3)))
    return y ** 2


def wave200(z, x):
    sigma = z * 4 * x / a0
    y = np.exp(-sigma / 2) * np.sqrt(z ** 3 / (32 * np.pi * (a0 ** 3))) * (
                2 - sigma)
    return y ** 2


def wave300(z, x):
    sigma = z * 4 * x / a0
    y = np.exp(-sigma / 3) * np.sqrt(
        z ** 3 / (81 * 81 * 3 * np.pi * (a0 ** 3))) * (
                27 - 18 * sigma + 2 * sigma ** 2)
    return y ** 2


def wave(z, x):
    y = wave100(z, x)
    y += 4 * wave200(z, x)
    # y += 3 * wave300(z, x)
    return 4 * np.pi * ((2 * x) ** 2) * y


def gauss(x, sigma=1):
    return np.exp(-2 * x / (2 * sigma ** 2))


def main():
    x = np.arange(0, 16, 0.1)
    # wave100(z=7)
    # wave100(z=8)
    y1 = wave(10, x)
    # print(y1)
    plt.plot(x, y1)
    plt.show()
    # y2 = wave(8, 4.9 - x)
    # print(y2)
    # print(y1 + y2)
    # plt.plot(x, y2)
    # plt.show()
    # plt.plot(x, y1 + y2)
    # plt.show()

    # y = gauss(x, 1)
    # plt.plot(x, y ** 0.2)
    # plt.show()


if __name__ == '__main__':
    main()
