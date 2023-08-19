# %%
import math
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from keras.applications.inception_v3 import InceptionV3

from PIL import Image

import importlib

import control_panel as cp
import analysis as an
import utils.paths as paths
import utils.ploters as ploters 
import filters
import utils.bw_converter as bw
import models.vae as v1
import utils.image_generator as im_gen

# %%
importlib.reload(cp)
importlib.reload(an)
importlib.reload(paths)
importlib.reload(ploters)
importlib.reload(filters)
importlib.reload(bw)
importlib.reload(v1)
importlib.reload(im_gen)
importlib.reload(im_gen)


df = pd.read_csv(paths.getDataSetPath('styles.csv'))


# %%
an.showDatasetDetails(df)


# %% Plot random images from dataframe
importlib.reload(ploters)
ids = df['id']
selected_image_ids = random.sample(ids.tolist(),10)
ploters.plotImagesById(selected_image_ids)

# %% Plot only random whatches
importlib.reload(filters)
watches_df = filters.getWatches(df)
ploters.plotRandomImg(watches_df)


# %% WARNING: this code create a new folder with only grayscale watches
importlib.reload(ploters)
importlib.reload(bw)
bw.convert_to_bw(watches_df['id'])
ploters.plotRandomImg(watches_df, path=paths.BW_IMG_FOLDER_INNER)


# %% Show first image of a directory
import pathlib
data_dir = pathlib.Path(paths.BW_IMG_FOLDER_INNER).with_suffix('')
image_count = len(list(data_dir.glob('*.jpg')))
print(image_count)

l = list(data_dir.glob('*'))
print(l[0])

Image.open(str(l[0]))

# %% Generator
importlib.reload(im_gen)

train_data_generator, validation_data_generator = im_gen.createImageGenerator(paths.BW_IMG_FOLDER)
im_gen.plotGeneratedImages(train_data_generator)


# %% VAE
importlib.reload(v1)
import models.flatVAE as v2
importlib.reload(v2)
vae, vae_encoder, vae_decoder = v2.FlatVAE().getVAE(4800)
v1.plotVAE(vae)

kl_coefficient=1

#Information needed to compute the loss function
vae_input=vae.input
vae_output=vae.output
mu=vae.get_layer('mu').output
log_var=vae.get_layer('log_var').output

vae.add_loss(v1.vae_loss(vae_input,vae_output,mu,log_var,kl_coefficient,4800))
vae.compile(optimizer='adam')


loss_metrics=[]
val_metrics=[]
# %% This code do not work
epoch_count = 100
# batch_size=100
patience=5

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

history = vae.fit(generator,epochs=epoch_count,callbacks=[early_stop])


# %% =============TRAINING====================
epochs = 50
batch_size = 16

for epoch in range(epochs):
    
    train_x, tain_label_y = next(train_data_generator)  # Assuming the generator yields (batch_x, label_y)
    validation_x, validation_label_y = next(validation_data_generator)  # Assuming the generator yields (batch_x, label_y)
        
    wathes = np.reshape(train_x, (len(train_x),4800))
    watches_val = np.reshape(validation_x, (len(validation_x),4800))

    history = vae.fit(
        wathes, 
        y=wathes, 
        validation_data=(watches_val, watches_val), 
        batch_size=batch_size, 
        verbose=0)
   
    loss = history.history["loss"][0]
    val_loss = history.history["val_loss"][0]
    loss_metrics.append(loss)
    val_metrics.append(val_loss)
    print(f"Epoch {epoch+1}/{epochs} | loss: {loss} | val: {val_loss}")
print("end")


train_x = next(validation_data_generator)
wathes = train_x[0][:5]
print(wathes.shape)
ploters.plot_generated_images([wathes], 1, 5)
wathes = np.reshape(wathes, (len(wathes),4800))
result = vae.predict(wathes)
result = np.reshape(result, (len(result), 80, 60, 1))
ploters.plot_generated_images([result], 1, 5)

history.history["loss"] = loss_metrics
history.history["val_loss"] = val_metrics
ploters.plot_history(history)

# %%
encoder_input_size = vae_decoder.layers[0].input_shape[0][1]
num_images = 20
images_in_cols = 6
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
    generated = decoded_x.reshape(len(decoded_x), 80, 60, 1)
    generated_images.append(generated)      

ploters.plot_generated_images(generated_images,rows,images_in_cols)



# %% Works Only for 2D latent space
n = 15 # number of images per row and column
limit=3 # random values are sampled from the range [-limit,+limit]

grid_x = np.linspace(-limit,limit, n) 
grid_y = np.linspace(limit,-limit, n)

generated_images=[]
for i, yi in enumerate(grid_y):
  single_row_generated_images=[]
  for j, xi in enumerate(grid_x):
    random_sample = np.array([[ xi, yi]])
    decoded_x = vae_decoder.predict(random_sample,verbose=0)
    single_row_generated_images.append(decoded_x[0].reshape(80, 60, 1))
  generated_images.append(single_row_generated_images)      

ploters.plot_generated_images(generated_images,n,n,True)


