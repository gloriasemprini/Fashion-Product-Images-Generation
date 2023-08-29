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

import vae_models.vae as v1
import vae_models.flatVAE as flat_vae
import vae_models.conVAE as conv_vae

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
NUM_COLORS = 3
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

num_images = 20
images_in_cols = 5
rows = math.ceil(num_images/images_in_cols)

decoderGen = im_gen.DecoderImageGenerator(vae_decoder, images_in_cols)
iterator = iter(decoderGen)

generated_images=[]
for row in range(rows):
    generated_images.append(next(iterator))      

ploters.plot_generated_images(generated_images,rows,images_in_cols)

if(latent_space_dimansion == 2):
   ploters.plot_2d_latent_space(vae_decoder, image_shape)




# %% FID metric
from tqdm import tqdm
inception_model = tf.keras.applications.InceptionV3(include_top=False, 
                            input_shape=image_shape,
                            weights="imagenet", 
                            pooling='avg')
def compute_embeddings(dataloader, count):
    image_embeddings = []
    for _ in range(count):
        images = next(iter(dataloader))
        embeddings = inception_model.predict(images)
        image_embeddings.extend(embeddings)
    return np.array(image_embeddings)


count = 100 # math.ceil(10000/BATCH_SIZE)


# compute embeddings for real images
real_image_embeddings = compute_embeddings(train_generator, count)


image_generator = im_gen.DecoderImageGenerator(vae_decoder, BATCH_SIZE)
# compute embeddings for generated images
generated_image_embeddings = compute_embeddings(iter(image_generator), count)


print("Real embedding shape: " + str(real_image_embeddings.shape))
print("Generated embedding shape: " + str(generated_image_embeddings.shape))

def calculate_fid(real_embeddings, generated_embeddings):
    # calculate mean and covariance statistics
    mu1, sigma1 = real_embeddings.mean(axis=0), np.cov(real_embeddings, rowvar=False)
    mu2, sigma2 = generated_embeddings.mean(axis=0), np.cov(generated_embeddings,  rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
        # calculate score
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid


fid = calculate_fid(real_image_embeddings, generated_image_embeddings)

print("FID: " + str(fid))
# %%
