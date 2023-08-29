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

import utils.paths as paths
import utils.callbacks as callbacks
import utils.ploters as ploters 

import utils.image_generator as im_gen

import gan.gan as g1
import utils.gan_utils as g_ut

# %%
importlib.reload(im_gen)
importlib.reload(paths)
importlib.reload(ploters)
importlib.reload(g1)
# %% Generator
importlib.reload(im_gen)
image_heigh = 80
image_weigh = 80
NUM_COLORS = 1
BATCH_SIZE = 64
imageSize = (image_heigh, image_weigh)

train_generator, validation_generator = im_gen.createImageGenerator(
   # paths.BW_IMG_FOLDER, 
   paths.COLOR_IMG_FOLDER,
   imageSize=imageSize,
   batch_size=BATCH_SIZE,
   rgb=(NUM_COLORS==3))
#train_generator, validation_generator = im_gen.createImageGenerator(paths.COLOR_IMG_FOLDER, imageSize=imageSize)


im_gen.plotGeneratedImages(train_generator)

# %% GAN
importlib.reload(g1)


# %% Model creation
input_noise_dim=100
NUM_PIXELS = image_heigh * image_weigh * NUM_COLORS
image_shape = (image_heigh, image_weigh, NUM_COLORS)
arr = [256,512,1024]

gan, gan_generator, gan_discriminator = g1.Gan().build_gan(
  input_noise_dim,
  arr,
  image_shape,
  'relu',
  'sigmoid')

# %% Summary
gan.summary()
keras.utils.plot_model(gan,show_shapes=True, show_layer_names=True,expand_nested=True)

# %% Model compilation
gan_discriminator.compile(loss='binary_crossentropy', optimizer='sgd')

gan_discriminator.trainable = False
gan.compile(loss='binary_crossentropy', optimizer='sgd')




# %% Execute training
epoch_count=10
batch_size=100

d_epoch_losses,g_epoch_losses=g1.train_gan(gan,
                                        gan_generator,
                                        gan_discriminator,
                                        g_ut.val_x_flatten,
                                        g_ut.val_x_flatten.shape[0],
                                        input_noise_dim,
                                        epoch_count,
                                        batch_size,
                                        g_ut.get_gan_random_input,
                                        g_ut.get_gan_real_batch,
                                        g_ut.get_gan_fake_batch,
                                        g_ut.concatenate_gan_batches,
                                        plt_frq=1,
                                        plt_example_count=15)

ploters.plot_gan_losses(d_epoch_losses,g_epoch_losses)

# %% Tips and tricks for training
train_x_flatten = (train_x_flatten*2)-1
val_x_flatten = (val_x_flatten*2)-1
test_x_flatten = (test_x_flatten*2)-1

generator_output_activation='tanh'

# $$ Avoid sparse gradients
hidden_activation=keras.layers.LeakyReLU(alpha=0.2)

# %% One-sided label smoothing
use_one_sided_labels=True

# %% Use Adam optimizer
optimizer = keras.optimizers.legacy.Adam(learning_rate=0.0002, beta_1=0.5)

# %% Execute training
epoch_count=30
batch_size=100

gan,gan_generator,gan_discriminator=g1.build_gan(input_noise_dim,
                                              [256,512,1024],
                                              train_x_flatten.shape[1],
                                              hidden_activation,
                                              generator_output_activation)

gan_discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)

gan_discriminator.trainable = False
gan.compile(loss='binary_crossentropy', optimizer=optimizer)

d_epoch_losses,g_epoch_losses=g1.train_gan(gan,
                                        gan_generator,
                                        gan_discriminator,
                                        val_x_flatten,
                                        val_x_flatten.shape[0],
                                        input_noise_dim,
                                        epoch_count,
                                        batch_size,
                                        g_ut.get_gan_random_input,
                                        g_ut.get_gan_real_batch,
                                        g_ut.get_gan_fake_batch,
                                        g_ut.concatenate_gan_batches,
                                        use_one_sided_labels=use_one_sided_labels,
                                        plt_frq=1,
                                        plt_example_count=15)

ploters.plot_gan_losses(d_epoch_losses,g_epoch_losses)

# %% Generated images
noise = np.random.normal(0, 1, size=(1, input_noise_dim))

generated_x = gan_generator.predict(noise,verbose=0)
digit = generated_x[0].reshape(original_image_shape)

plt.axis('off')
plt.imshow(digit, cmap='gray')
plt.show()

# %% RUN
n = 10 # number of images per row and column

generated_images=[]
for i in range(n):
  noise = np.random.normal(0, 1, size=(n, input_noise_dim))
  generated_x = gan_generator.predict(noise,verbose=0)
  generated_images.append([g.reshape(original_image_shape) for g in generated_x])

ploters.plot_generated_images(generated_images,n,n)