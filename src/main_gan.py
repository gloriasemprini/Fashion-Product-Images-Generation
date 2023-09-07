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
import gan.dcgan as dcg1
import utils.gan_utils as g_ut

# %%
importlib.reload(img_gen)
importlib.reload(paths)
importlib.reload(ploters)
importlib.reload(g1)
importlib.reload(cg1)
importlib.reload(g_ut)
# %% Generator
importlib.reload(img_gen)
image_heigh = 64
image_weigh = 64

NUM_COLORS = 1
BATCH_SIZE = 64
imageSize = (image_heigh, image_weigh)

train_generator, validation_generator = img_gen.createImageGenerator(
   # paths.BW_IMG_FOLDER, 
   paths.COLOR_IMG_FOLDER,
   imageSize=imageSize,
   batch_size=BATCH_SIZE,
   rgb=(NUM_COLORS==3))
#train_generator, validation_generator = im_gen.createImageGenerator(paths.COLOR_IMG_FOLDER, imageSize=imageSize)


img_gen.plotGeneratedImages(train_generator)

# %% GAN - Model creation
importlib.reload(g1)
importlib.reload(g_ut)

input_noise_dim=100
NUM_PIXELS = image_heigh * image_weigh * NUM_COLORS
image_shape = (image_heigh, image_weigh, NUM_COLORS)
arr = [256,512,1024]

gan, gan_generator, gan_discriminator = g1.Gan().build_gan(
                                        input_noise_dim,
                                        arr,
                                        image_shape,
                                        NUM_PIXELS,
                                        'relu',
                                        'sigmoid')

gan_generator.summary()
gan_discriminator.summary()
g_ut.plotGAN(gan)

gan_discriminator.compile(loss='binary_crossentropy', optimizer='sgd')

gan_discriminator.trainable = False
gan.compile(loss='binary_crossentropy', optimizer='sgd')




# %% Execute training
epoch_count=1000
batch_size=100

d_epoch_losses,g_epoch_losses=g1.Gan().train_gan(gan,
                                        gan_generator,
                                        gan_discriminator,
                                        train_generator,
                                        2000, # TODO numero di immagini
                                        input_noise_dim,
                                        epoch_count,
                                        batch_size,
                                        g_ut.get_gan_random_input,
                                        g_ut.get_gan_real_batch,
                                        g_ut.get_gan_fake_batch,
                                        g_ut.concatenate_gan_batches,
                                        plt_frq=20,
                                        plt_example_count=15,
                                        example_shape=image_shape)

ploters.plot_gan_losses(d_epoch_losses,g_epoch_losses)

# %% Tips prof, ma il normalize input non serve, no?
importlib.reload(g1)
importlib.reload(g_ut)

generator_output_activation='tanh'
use_one_sided_labels=True
optimizer = keras.optimizers.legacy.Adam(learning_rate=0.0002, beta_1=0.5)

NUM_PIXELS = image_heigh * image_weigh * NUM_COLORS
image_shape = (image_heigh, image_weigh, NUM_COLORS)
arr = [256,512,1024]

hidden_activation=layers.LeakyReLU(alpha=0.2)

gan,gan_generator,gan_discriminator=g1.Gan().build_gan(input_noise_dim,
                                              arr,
                                              image_shape,
                                              NUM_PIXELS,
                                              hidden_activation,
                                              generator_output_activation)


gan_generator.summary()
gan_discriminator.summary()
g_ut.plotGAN(gan)


gan_discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)

gan_discriminator.trainable = False
gan.compile(loss='binary_crossentropy', optimizer=optimizer)

# %%
epoch_count=100
batch_size=128

d_epoch_losses,g_epoch_losses=g1.Gan().train_gan(gan,
                                        gan_generator,
                                        gan_discriminator,
                                        train_generator,
                                        2000, # TODO numero di immagini
                                        input_noise_dim,
                                        epoch_count,
                                        batch_size,
                                        g_ut.get_gan_random_input,
                                        g_ut.get_gan_real_batch,
                                        g_ut.get_gan_fake_batch,
                                        g_ut.concatenate_gan_batches,
                                        use_one_sided_labels=use_one_sided_labels,
                                        plt_frq=10,
                                        plt_example_count=15,
                                        example_shape=image_shape)

ploters.plot_gan_losses(d_epoch_losses,g_epoch_losses)

# %%
"""importlib.reload(cg1)
importlib.reload(g_ut)

input_noise_dim=100
hidden_activation=layers.LeakyReLU(alpha=0.2)
generator_output_activation='tanh'

NUM_PIXELS = image_heigh * image_weigh * NUM_COLORS
image_shape = (image_heigh, image_weigh, NUM_COLORS)
arr = [256,512,1024]
#input, cond, arr, shape, n_pixel, hidd,gen
cg1.cGan().build_cgan
cgan,cgan_generator,cgan_discriminator=cg1.cGan().build_cgan(input_noise_dim,
                                                             arr,
                                                             image_shape,
                                                             NUM_PIXELS,
                                                             hidden_activation,
                                                             generator_output_activation,
                                                             train_generator.num_classes)
cgan.summary()
g_ut.plotGAN(cgan)

# %%
optimizer = keras.optimizers.legacy.Adam(learning_rate=0.0002, beta_1=0.5)

cgan_discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)

cgan_discriminator.trainable = False
cgan.compile(loss='binary_crossentropy', optimizer=optimizer)

# %%
epoch_count=50
batch_size=100

d_epoch_losses,g_epoch_losses=train_gan(cgan,
                                        cgan_generator,
                                        cgan_discriminator,
                                        [val_x_flatten,val_y_one_hot],
                                        val_x_flatten.shape[0],
                                        input_noise_dim,
                                        epoch_count,
                                        batch_size,
                                        get_cgan_random_input,
                                        get_cgan_real_batch,
                                        get_cgan_fake_batch,
                                        concatenate_cgan_batches,
                                        condition_count=category_count,
                                        use_one_sided_labels=True,
                                        plt_frq=1,
                                        plt_example_count=15)

plot_gan_losses(d_epoch_losses,g_epoch_losses) """
# %%

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
      with tf.device('/device:GPU:0'):
         selected_gpu = gpus[0]  # Seleziona la prima GPU
         print(f"Stai usando la GPU: {selected_gpu.name}")

# %% DCGAN
importlib.reload(dcg1)
importlib.reload(g1)
importlib.reload(g_ut)

with tf.device('/device:GPU:0'):

   #input_noise_dim=100
   input_noise_dim=100

   NUM_PIXELS = image_heigh * image_weigh * NUM_COLORS
   image_shape = (image_heigh, image_weigh, NUM_COLORS)

   dcgan,dcgan_generator,dcgan_discriminator=dcg1.dcGan().build_dcgan(input_noise_dim)
   #dcgan.summary()
   dcgan_generator.summary()
   dcgan_discriminator.summary()
   g_ut.plotdcGAN(dcgan)
   optimizer = keras.optimizers.Adam(clipnorm=0.01, learning_rate=0.0002, beta_1=0.8)
   #optimizer_a = keras.optimizers.legacy.RMSprop()
   dcgan_discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)

   dcgan_discriminator.trainable = False
   dcgan.compile(loss='binary_crossentropy', optimizer=optimizer)

# %%

epoch_count=20
#batch_size=32
batch_size=64
#batch_size=128
with tf.device('/device:GPU:0'):
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
