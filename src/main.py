# %%
import math
import numpy as np
import matplotlib.pyplot as plt
import random
from tensorflow import keras
from numba import cuda
import importlib

import utils.paths as paths
import utils.callbacks as callbacks
import utils.ploters as ploters 
import models.vae as v1
import utils.image_generator as im_gen


# %%
importlib.reload(v1)
importlib.reload(im_gen)
importlib.reload(paths)
importlib.reload(ploters)



# %% Generator
importlib.reload(im_gen)
image_heigh = 80
image_weigh = 80
imageSize = (image_heigh, image_weigh)

train_generator, validation_generator = im_gen.createImageGenerator(
   paths.BW_IMG_FOLDER, 
   imageSize=imageSize,
   batch_size=64)
#train_generator, validation_generator = im_gen.createImageGenerator(paths.COLOR_IMG_FOLDER, imageSize=imageSize)


im_gen.plotGeneratedImages(train_generator)


# %% VAE
importlib.reload(v1)
import models.flatVAE as v2
import models.conVAE as v3
importlib.reload(v2)
importlib.reload(v3)
NUM_COLORs = 1
NUM_PIXELS = image_heigh * image_weigh * NUM_COLORs
image_shape = (image_heigh, image_weigh, NUM_COLORs)
latent_space_dimansion = 2
vae, vae_encoder, vae_decoder = v3.ConvVae().build_vae(
   image_shape, 
   NUM_PIXELS, 
   [1024, 256], 
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


loss_metrics=[]
val_metrics=[]


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
encoder_input_size = vae_decoder.layers[0].input_shape[0][1]
num_images = 20
images_in_cols = 5
rows = math.ceil(num_images/images_in_cols)

generated_images=[]
for row in range(rows):
    single_row_generated_images=[]
    for col in range(images_in_cols):
        random_sample = []
        for i in range(encoder_input_size):
            random_sample.append(random.normalvariate(0,1))
        single_row_generated_images.append(random_sample)
    single_row_generated_images = np.array(single_row_generated_images)
    decoded_x = vae_decoder.predict(single_row_generated_images,verbose=0)
    generated = decoded_x.reshape(len(decoded_x), image_heigh, image_weigh, NUM_COLORs)
    generated_images.append(generated)      

ploters.plot_generated_images(generated_images,rows,images_in_cols)

if(latent_space_dimansion == 2):
   ploters.plot_2d_latent_space(vae_decoder, image_shape)


