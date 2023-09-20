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
from keras.utils import to_categorical
import utils.df_preprocessing as preprocess

# %%
importlib.reload(img_gen)
importlib.reload(paths)
importlib.reload(ploters)
importlib.reload(g1)
importlib.reload(cg1)
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
 
CLASSES = ["Sunglasses"]
# %% DF Generator
importlib.reload(img_gen)
importlib.reload(preprocess)

#parameters
BATCH_SIZE = 128
image_heigh = 64
image_weigh = 64
num_color_dimensions = 3 # 1 for greyscale or 3 for RGB
with_color_label = False # class label inlude article color

# Computed parameters
image_size = (image_heigh, image_weigh)
image_shape = (image_heigh, image_weigh, num_color_dimensions)
num_pixels = image_heigh * image_weigh * num_color_dimensions
rgb_on = (num_color_dimensions==3)
is_fid_active = image_heigh == image_weigh and image_weigh > 75 and rgb_on
if(with_color_label and (not rgb_on)): # error check
   raise Exception("Illegal state: color label can be used only with RGB images")

class_mode = "multi_output" if(with_color_label) else "categorical"
train_provider, val_provider  = img_gen.create_data_provider_df(
    paths.IMG_FOLDER,
    CLASSES,
    class_mode=class_mode,
    image_size=image_size,
    batch_size=BATCH_SIZE,
    rgb=rgb_on,
)
one_hot_label_len = train_provider.num_classes if(with_color_label) else len(train_provider.class_indices)
if(type(train_provider) is img_gen.MultiLabelImageDataGenerator):
    all_one_hot_labels = train_provider.labels
else:
    all_one_hot_labels = to_categorical(train_provider.labels)

img_gen.plot_provided_images(train_provider)

# %% Tips prof, ma il normalize input non serve, no?
importlib.reload(g1)
importlib.reload(g_ut)

generator_output_activation='tanh'
use_one_sided_labels=True
optimizer = keras.optimizers.legacy.Adam(learning_rate=0.0002, beta_1=0.5)

input_noise_dim=100
arr = [256,512,1024]

hidden_activation=layers.LeakyReLU(alpha=0.2)

gan,gan_generator,gan_discriminator=g1.Gan().build_gan(input_noise_dim,
                                              arr,
                                              image_shape,
                                              num_pixels,
                                              hidden_activation,
                                              generator_output_activation)

gan_generator.summary()
gan_discriminator.summary()
g_ut.plotGAN(gan)


gan_discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)

gan_discriminator.trainable = False
gan.compile(loss='binary_crossentropy', optimizer=optimizer)

# %%
epoch_count=50

d_epoch_losses,g_epoch_losses=g1.Gan().train_gan(gan,
                                        gan_generator,
                                        gan_discriminator,
                                        train_provider,
                                        2000, # TODO numero di immagini
                                        input_noise_dim,
                                        epoch_count,
                                        BATCH_SIZE,
                                        g_ut.get_gan_random_input,
                                        g_ut.get_gan_real_batch,
                                        g_ut.get_gan_fake_batch,
                                        g_ut.concatenate_gan_batches,
                                        use_one_sided_labels=use_one_sided_labels,
                                        plt_frq=5,
                                        plt_example_count=15,
                                        image_shape=image_shape)

ploters.plot_gan_losses(d_epoch_losses,g_epoch_losses)


# %% CGAN
