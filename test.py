from PIL import Image
import gabor
import feature
from matplotlib import pyplot as plt
import time
import math
import matcher
import os
import numpy as np

data_path = 'D:\\palmprint_dataset\\polyu_v2'
result_path = 'D:\\palmprint_dataset\\polyu_v2'
file_names = []
palms_per_person = 50
persons = 600

# im = Image.open('test0.bmp')
# filter = gabor.single_gabor(35,4.6,2.6,0)
# feature.fingle_convole_slow(im,filter)

for i in range(1, persons + 1):
    for j in range(1, palms_per_person + 1):
        file_name = '{}\\{:0>4d}_{:0>2d}.bmp'.format(data_path, i, j)

        if os.path.exists(file_name):
            file_names.append(file_name)

# generate filters
filters = gabor.generate()

# distance matrix
dist_matrix = np.zeros([len(file_names), len(file_names)])

for i in range(0, len(file_names) - 1):
    im1 = Image.open(file_names[i])
    f1 = feature.extract(im1, filters)
    m1 = matcher.get_mask(im1)

    # start = time.clock()
    for j in range(i + 1, len(file_names)):
        im2 = Image.open(file_names[j])

        start = time.clock()

        f2 = feature.extract(im2, filters)
        m2 = matcher.get_mask(im2)

        d = matcher.compute_distance(f1, f2, m1, m2, 4)

        dist_matrix[i][j] = d
        dist_matrix[j][i] = dist_matrix[i][j]

        print(time.clock() - start)
    # print(time.clock() - start)


# np.savetxt('data.csv', dist_matrix, delimiter=',')
save_file_name = '{}\\dist.sim'.format(result_path)
fo = open(save_file_name,'wb')

fo.write(dist_matrix) # type float64 if using numpy ndarray

fo.close()