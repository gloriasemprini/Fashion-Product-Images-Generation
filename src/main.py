# %%
import math
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy import linalg
import tensorflow as tf
from tensorflow import keras
from numba import cuda
from tqdm import tqdm
import importlib

import utils.paths as paths
import utils.callbacks as callbacks
import utils.ploters as ploters 

import vae_models.vae as v1
import vae_models.flatVAE as flat_vae
import vae_models.conVAE as conv_vae

import utils.image_generator as img_gen
import metrics.fid as fid

# %%
importlib.reload(v1)
importlib.reload(img_gen)
importlib.reload(paths)
importlib.reload(ploters)
importlib.reload(fid)



# %% Generator
importlib.reload(img_gen)
image_heigh = 80
image_weigh = 80
NUM_COLORS = 3
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


# %% VAE
importlib.reload(v1)
importlib.reload(flat_vae)
importlib.reload(conv_vae)


NUM_PIXELS = image_heigh * image_weigh * NUM_COLORS
image_shape = (image_heigh, image_weigh, NUM_COLORS)
latent_space_dimansion = 128
vae, vae_encoder, vae_decoder = conv_vae.ConvVae().build_vae(
   image_shape, 
   NUM_PIXELS, 
   [1024], 
   latent_space_dimansion,
   'LeakyReLU',
   'sigmoid')
vae.summary()
v1.plotVAE(vae)

kl_coefficient=1

#Information needed to compute the loss function
vae_input=vae.input
vae_output=vae.output
mu=vae.get_layer('mu').output
log_var=vae.get_layer('log_var').output

vae.add_loss(v1.vae_loss(vae_input,vae_output,mu,log_var,kl_coefficient,NUM_PIXELS))
vae.compile(optimizer='adam', run_eagerly=True)


# loss_metrics=[]
# val_metrics=[]


# %%  ============= Automatic TRAINING====================
epoch_count = 5
patience=10

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
intermediateImages = callbacks.CustomFitCallback(validation_generator, epochs_per_fit=10)

history = vae.fit(
   train_generator, 
   epochs=epoch_count,
   validation_data=validation_generator,
   callbacks=[early_stop, intermediateImages])

#Print loss chart
ploters.plot_history(history)

# Trasform 5 random images from validation set
train_x = next(validation_generator)
print(train_x.shape)
wathes = train_x[:5]
print(wathes.shape)
ploters.plot_generated_images([wathes], 1, 5)
result = vae.predict(wathes)
ploters.plot_generated_images([result], 1, 5)



# %% Autogenerate new images

num_images = 20
images_in_cols = 5
rows = math.ceil(num_images/images_in_cols)

decoderGen = img_gen.DecoderImageGenerator(vae_decoder, images_in_cols)
iterator = iter(decoderGen)

generated_images=[]
for row in range(rows):
    generated_images.append(next(iterator))      

ploters.plot_generated_images(generated_images,rows,images_in_cols)

if(latent_space_dimansion == 2):
   ploters.plot_2d_latent_space(vae_decoder, image_shape)





# %% Work only with RGB images
importlib.reload(fid)
image_generator = img_gen.DecoderImageGenerator(vae_decoder, BATCH_SIZE)
fid.getFid(train_generator, image_generator, image_shape)
