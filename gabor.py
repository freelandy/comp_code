import numpy as np
import math


def generate():
    angles = [0, math.pi / 6, 2 * math.pi / 6, 3 * math.pi / 6, 4 * math.pi / 6, 5 * math.pi / 6]
    filter_size = 35
    sigma = 4.6
    delta = 2.6

    filters = []
    for angle in angles:
        f = single_gabor(filter_size, sigma, delta, angle)
        filters.append(f)

    return filters


def single_gabor(filter_size, sigma, delta, angle):
    half_filter_size = math.floor(filter_size / 2)
    ln22 = math.sqrt(2 * math.log(2))
    k = ln22 * (delta + 1) / (delta - 1)
    omega = k / sigma

    factor1 = -omega / (math.sqrt(2 * math.pi) * k)
    factor2 = -(omega * omega) / (8 * k * k)

    [y, x] = np.meshgrid(range(-half_filter_size, half_filter_size + 1), range(-half_filter_size, half_filter_size + 1))

    x1 = np.multiply(x, math.cos(angle)) + np.multiply(y, math.sin(angle))
    y1 = np.multiply(y, math.cos(angle)) - np.multiply(x, math.sin(angle))

    t = np.multiply(np.exp(np.multiply(np.add(np.multiply(np.square(x1), 4), np.square(y1)), factor2)), factor1)
    r = np.multiply(np.cos(np.add(np.multiply(x, omega * math.cos(angle)), np.multiply(y, omega * math.sin(angle)))), t)

    m = np.mean(r)
    r = np.add(r, -m)

    return r
