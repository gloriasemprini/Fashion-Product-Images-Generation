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
import vae_models.convVAE as conv_vae
import vae_models.cConvVAE as cconv_vae

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
image_heigh = 64
image_weigh = 64
NUM_COLORS = 3
BATCH_SIZE = 64
imageSize = (image_heigh, image_weigh)

train_generator, validation_generator = img_gen.createImageGenerator(
   paths.COLOR_IMG_FOLDER,
   imageSize=imageSize,
   batch_size=BATCH_SIZE,
   rgb=(NUM_COLORS==3),
   class_mode='categorical')



img_gen.plotGeneratedImages(train_generator)

# %%
q = next(train_generator)

# %% VAE
importlib.reload(v1)
importlib.reload(flat_vae)
importlib.reload(conv_vae)
importlib.reload(cconv_vae)


NUM_PIXELS = image_heigh * image_weigh * NUM_COLORS
image_shape = (image_heigh, image_weigh, NUM_COLORS)
latent_space_dimansion = 256
vae, vae_encoder, vae_decoder = cconv_vae.cConvVae().build_vae(
   image_shape, 
   NUM_PIXELS, 
   [2048, 512], 
   latent_space_dimansion,
   'LeakyReLU',
   'sigmoid',
   train_generator.num_classes)

vae.summary()
v1.plotVAE(vae)

### Compilation
kl_coefficient=1
#Information needed to compute the loss function
vae_input=vae.input
vae_output=vae.output
mu=vae.get_layer('mu').output
log_var=vae.get_layer('log_var').output

vae.add_loss(v1.vae_loss(vae_input,vae_output,mu,log_var,kl_coefficient,NUM_PIXELS))
vae.compile(optimizer=tf.keras.optimizers.Adam(clipvalue=0.1), run_eagerly=True)


# loss_metrics=[]
# val_metrics=[]

# val_generator.shuffle = False
# val_generator.index_array = None


# %% Manual training
epoch_count = 50

def show(generator, model):
   # Trasform 5 random images from validation set
   val_x, val_y = next(generator)
   if (len(val_x) < 5):
      val_x, val_y = next(generator)

   # get first 5 dataset images
   watches = val_x[:5] 
   labels = val_y[:5]
   ploters.plot_generated_images([watches], 1, 5)

   generated_watches = model.predict([watches,labels])
   ploters.plot_generated_images([generated_watches], 1, 5)

for e in range(1, epoch_count+1):
   avg_loss = 0
   for i in range(len(train_generator)):
      train_x, train_y = next(train_generator)
      # val_x, val_y = next(validation_generator)
      loss = vae.train_on_batch([train_x, train_y], train_x)
      print(".", end="")
      current_batch_size = len(train_x)
      avg_loss+=loss*current_batch_size
   print("x avg_loss", avg_loss/(len(train_generator)*current_batch_size))

   if(e%5 == 0):
      print("current epoch is ",e)
      show(validation_generator, vae)

show(validation_generator, vae)

# %%  ============= Automatic TRAINING====================
epoch_count = 2
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

# get first 5 dataset images
watches = train_x[0][:5] if(type(train_x) is tuple) else train_x[:5]
ploters.plot_generated_images([watches], 1, 5)
generated_watches = vae.predict(watches)
ploters.plot_generated_images([generated_watches], 1, 5)



# %% Autogenerate new images

num_images = 20
images_in_cols = 5
rows = math.ceil(num_images/images_in_cols)

# decoderGen = img_gen.ImageGeneratorDecoder(vae_decoder, images_in_cols)
one_hot = np.zeros(train_generator.num_classes, dtype=float)
one_hot[0] = 1.0
decoderGen = img_gen.ConditionalImageGeneratorDecoder(vae_decoder, images_in_cols,label=one_hot)
iterator = iter(decoderGen)

generated_images=[]
for row in range(rows):
    generated_images.append(next(iterator))      

ploters.plot_generated_images(generated_images,rows,images_in_cols)

if(latent_space_dimansion == 2):
   ploters.plot_2d_latent_space(vae_decoder, image_shape)



# %% Work only with RGB images
importlib.reload(fid)
image_generator = img_gen.ImageGeneratorDecoder(vae_decoder, BATCH_SIZE)
fid.getFid(train_generator, image_generator, image_shape)
