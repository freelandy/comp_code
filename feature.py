from PIL import Image
from PIL import ImageFilter
from matplotlib import pyplot as plt
import numpy as np
import scipy as sp
from scipy import signal


def extract(im, filters):
    res = []
    idx = 0
    for f in filters:
        # r = single_convolve_slow(im, f)
        r = single_convolve(im, f)
        res.append(r)

    max_idx = np.argmax(res, 0) # 0 is the 3rd dimension

    return max_idx


def encode(max_idx):
    pass

    return


def single_convolve(im, filter):
    # convolve im with filter
    # im must be 128*128
    # filter must be 35*35

    im_array = np.array(im)
    im_array = im_array / 255

    step = 3
    margin = 17
    half_filter_size = 17

    r = np.zeros([32, 32])
    x_idx = 0
    for x in range(0 + margin - 1, 128 - margin + 1, step):
        y_idx = 0
        for y in range(0 + margin - 1, 128 - margin + 1, step):
            # crop a small patch around (x,y)
            patch = im_array[x - half_filter_size + 1:x + half_filter_size + 1 + 1, y - half_filter_size + 1:y + half_filter_size + 1 + 1]

            # convolve
            r[x_idx, y_idx] = np.sum(np.multiply(patch, filter))
            y_idx = y_idx + 1

        x_idx = x_idx + 1

    return r


def single_convolve_slow(im, filter):
    im_array = np.array(im)
    im_array = im_array / 255

    filtered_array = sp.signal.convolve2d(im_array,filter,boundary='symm',mode='same')

    step = 3
    margin = 17
    r = filtered_array[0 + margin - 1:128 - margin + 1:step,0 + margin - 1:128 - margin + 1:step]

    # just for debug
    # plt.figure()
    # plt.imshow(r)
    # plt.show()

    return r