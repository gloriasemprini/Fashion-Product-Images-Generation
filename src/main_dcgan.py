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
import gan.dcgan as dcg1
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
importlib.reload(dcg1)
importlib.reload(g1)
importlib.reload(g_ut)

#input_noise_dim=100
input_noise_dim=100

NUM_PIXELS = image_heigh * image_weigh * NUM_COLORS
image_shape = (image_heigh, image_weigh, NUM_COLORS)

dcgan,dcgan_generator,dcgan_discriminator=dcg1.dcGan().build_dcgan(input_noise_dim)
#dcgan.summary()
dcgan_generator.summary()
dcgan_discriminator.summary()
g_ut.plotdcGAN(dcgan)
optimizer = keras.optimizers.Adam(clipnorm=0.01, learning_rate=0.000005, beta_1=0.8)
#optimizer_a = keras.optimizers.legacy.RMSprop()
dcgan_discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)

dcgan_discriminator.trainable = False
dcgan.compile(loss='binary_crossentropy', optimizer=optimizer)

# %%

epoch_count=200

d_epoch_losses,g_epoch_losses=g1.Gan().train_gan(dcgan,
                                        dcgan_generator,
                                        dcgan_discriminator,
                                        train_generator,
                                        2000,
                                        input_noise_dim,
                                        epoch_count,
                                        batch_size,
                                        g_ut.get_gan_random_input,
                                        g_ut.get_gan_real_batch,
                                        g_ut.get_gan_fake_batch,
                                        g_ut.concatenate_gan_batches,
                                        use_one_sided_labels=True,
                                        plt_frq=2,
                                        plt_example_count=15,
                                        example_shape=image_shape)

ploters.plot_gan_losses(d_epoch_losses,g_epoch_losses)
# %%
