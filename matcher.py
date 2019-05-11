import numpy as np
import math


def compute_distance(f1, f2, m1, m2, displacement):
    ds = []
    f2_ext = np.zeros([32 + 2 * displacement, 32 + 2 * displacement])
    f2_ext[0 + displacement:31 + displacement + 1, 0 + displacement:31 + displacement + 1] = f2

    m2_ext = np.zeros([32 + 2 * displacement, 32 + 2 * displacement])
    m2_ext[0 + displacement:31 + displacement + 1, 0 + displacement:31 + displacement + 1] = m2

    for x in range(-displacement, displacement + 1):
        for y in range(-displacement, displacement + 1):
            f1_ext = np.zeros([32 + 2 * displacement, 32 + 2 * displacement])
            f1_ext[0 + x + displacement:31 + x + displacement + 1, 0 + y + displacement:31 + y + displacement + 1] = f1

            m1_ext = np.zeros([32 + 2 * displacement, 32 + 2 * displacement])
            m1_ext[0 + x + displacement:31 + x + displacement + 1, 0 + y + displacement:31 + y + displacement + 1] = m1

            d = compute_distance_one_displacement(f1_ext, f2_ext, m1_ext, m2_ext)
            ds.append(d)

    ret = min(ds)

    return ret


def compute_distance_one_displacement(f1, f2, m1, m2):
    d = np.abs(np.add(f1, np.multiply(f2, -1)))
    nd = np.add(np.multiply(d, -1), 6)
    same_mat = np.minimum(d,nd)
    m = np.multiply(m1, m2)

    # in case that two masks are all zero!
    sum_m = np.sum(m)
    if sum_m != 0:
        ret = np.sum(np.multiply(same_mat, m)) / sum_m
    else:
        ret = 3

    return ret / 3


def get_mask(im):
    filter_size = 35
    half_filter_size = math.floor(filter_size / 2)
    margin = 17
    step = 3

    im_array = np.array(im)
    im_array = im_array / 255

    back_value_thres = 50 / 255
    back_count_thres = (60 / (35 * 35)) * (filter_size * filter_size)

    mask = np.ones([32, 32])
    x_idx = 0
    for x in range(0 + margin - 1, 128 - margin + 1, step):
        y_idx = 0
        for y in range(0 + margin - 1, 128 - margin + 1, step):
            # crop a small patch around (x,y)
            patch = im_array[x - half_filter_size + 1:x + half_filter_size + 1 + 1,
                    y - half_filter_size + 1:y + half_filter_size + 1 + 1]

            # count forground pixels
            cnt = np.sum(np.less(patch, back_value_thres))
            if cnt >= back_count_thres:
                mask[x_idx, y_idx] = 0

            y_idx = y_idx + 1

        x_idx = x_idx + 1

    return mask
