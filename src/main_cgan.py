# %%
import math
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy import linalg
import tensorflow as tf
from tensorflow import keras
from numba import cuda
import importlib
from keras import layers

import graphviz

import utils.paths as paths
import utils.callbacks as callbacks
import utils.ploters as ploters 

import utils.image_generator as img_gen

import gan.gan as g1
import gan.cgan as cg1
import utils.gan_utils as g_ut
# %%
importlib.reload(img_gen)
importlib.reload(paths)
importlib.reload(ploters)
importlib.reload(g1)
importlib.reload(g_ut)

# %%
#### Possible classes:

# "Watches" #2542 !
# "Handbags" #1759 !
# "Sunglasses" #1073 !
# "Belts" #813 !
# "Backpacks" #724
# "Sarees" #475
# "Deodorant" #347
# "Nail Polish" #329
# "Ties" #263

# "Sports Shoes" #2036
# "Flip Flops" #916 !
# "Formal Shoes" #637
 
CLASSES = ["Watches"]

# %% Generator
importlib.reload(img_gen)
image_heigh = 64
image_weigh = 64

NUM_COLORS = 1
batch_size = 64
imageSize = (image_heigh, image_weigh)

train_generator, validation_generator = img_gen.createImageGenerator(
   # paths.BW_IMG_FOLDER, 
   paths.COLOR_IMG_FOLDER,
   imageSize=imageSize,
   batch_size=batch_size,
   rgb=(NUM_COLORS==3))
#train_generator, validation_generator = im_gen.createImageGenerator(paths.COLOR_IMG_FOLDER, imageSize=imageSize)

img_gen.plot_provided_images(train_generator)

# %% DCGAN
importlib.reload(cg1)
importlib.reload(g1)
importlib.reload(g_ut)

input_noise_dim=100

NUM_PIXELS = image_heigh * image_weigh * NUM_COLORS
image_shape = (image_heigh, image_weigh, NUM_COLORS)

hidden_activation=layers.LeakyReLU(alpha=0.2)
generator_output_activation='tanh'

cgan,cgan_generator,cgan_discriminator=cg1.cGan().build_cgan(input_noise_dim,
                                                  category_count,
                                                  [256,512,1024],
                                                  train_x_flatten.shape[1],
                                                  hidden_activation,
                                                  generator_output_activation)