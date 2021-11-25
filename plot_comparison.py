"""
Generates comparison gifs of two sequences of images in subfolder gifs
with base_filename1 and base_filename2
Each sequence has interp_time amount of images
"""

import imageio
# import torch
import os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import rc

# def comparison_gif(base_filenameA, base_filenameB, interp_time):
#     for i = 1: # in range(interp_time):
#         img_fileA = base_filenameA + "{}.png".format(i)
#         img_fileB = base_filenameB + "{}.png".format(i)

#         plt.figure(1)
#         plt.subplot(211)
#         plt.imshow(img1)

#         plt.subplot(212)
#         plt.imshow(img2)
#         plt.show()
#         imgs.append(imageio.imread(img_file))
#         os.remove(img_file) 
#     imageio.mimwrite(filename, imgs)


filename_base = '1traj'
filename_s = filename_base + '_s'
filename_r = filename_base + '_r'

plt.figure(1)
plt.subplot(121)
plt.imshow(imageio.imread(filename_s + '29.png'))
plt.axis('off')
plt.subplot(122)
plt.imshow(imageio.imread(filename_r + '29.png'))
plt.axis('off')
# [axi.set_axis_off() for axi in ax.ravel()]
plt.show()