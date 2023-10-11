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

# import graphviz

import utils.paths as paths
import utils.callbacks as callbacks
import utils.ploters as ploters 

import utils.image_provider as img_gen

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
 
CLASSES = ["Sunglasses", "Backpacks"]
# %% DF Generator
importlib.reload(img_gen)
importlib.reload(preprocess)

#parameters
BATCH_SIZE = 32
image_heigh = 64
image_weigh = 64
num_color_dimensions = 1 # 1 for greyscale or 3 for RGB
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
importlib.reload(cg1)
importlib.reload(g_ut)

generator_output_activation='tanh'
use_one_sided_labels=True

input_noise_dim=100
arr = [128,256,512,1024]

hidden_activation=layers.LeakyReLU(alpha=0.2)

cgan,cgan_generator,cgan_discriminator=cg1.cGan().build_cgan(input_noise_dim,
                                                         arr,
                                                         image_shape,
                                                         num_pixels,
                                                         hidden_activation,
                                                         generator_output_activation,
                                                         one_hot_label_len)

cgan_generator.summary()
cgan_discriminator.summary()
g_ut.plotcGAN(cgan)

optimizer = keras.optimizers.Adam(learning_rate=0.00002)

cgan_discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)

cgan_discriminator.trainable = False
cgan.compile(loss='binary_crossentropy', optimizer=optimizer)

# %%
importlib.reload(g1)
epoch_count=500

d_epoch_losses,g_epoch_losses=g1.Gan().train_gan(cgan,
                                        cgan_generator,
                                        cgan_discriminator,
                                        train_provider,
                                        2000, # TODO numero di immagini
                                        input_noise_dim,
                                        epoch_count,
                                        BATCH_SIZE,
                                        g_ut.get_cgan_random_input,
                                        g_ut.get_cgan_real_batch,
                                        g_ut.get_cgan_fake_batch,
                                        g_ut.concatenate_cgan_batches,
                                        condition_count=one_hot_label_len,
                                        use_one_sided_labels=use_one_sided_labels,
                                        plt_frq=5,
                                        plt_example_count=15,
                                        image_shape=image_shape)

ploters.plot_gan_losses(d_epoch_losses,g_epoch_losses)
