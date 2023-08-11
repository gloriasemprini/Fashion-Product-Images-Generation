# %%
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
from tensorflow import keras
from keras import layers
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from PIL import Image

import importlib

import control_panel as cp
import analysis as an
import utils.paths as paths
import utils.ploters as ploters 
import filters
import utils.bw_converter as bw
import models.vae as v1

# %%
importlib.reload(cp)
importlib.reload(an)
importlib.reload(paths)
importlib.reload(ploters)
importlib.reload(filters)
importlib.reload(bw)
importlib.reload(v1)



df = pd.read_csv(paths.getDataSetPath('styles.csv'))


# %%
an.showDatasetDetails(df)


# %%
importlib.reload(ploters)
ids = df['id']
selected_image_ids = random.sample(ids.tolist(),10)
ploters.plotImagesById(selected_image_ids)

# %%
importlib.reload(filters)
watches_df = filters.getWatches(df)
ploters.plotRandomImg(watches_df)


# %%
importlib.reload(ploters)
importlib.reload(bw)
bw.convert_to_bw(watches_df['id'])
ploters.plotRandomImg(watches_df, path=paths.BW_IMG_FOLDER_INNER)


# %%
import pathlib
data_dir = pathlib.Path(paths.BW_IMG_FOLDER_INNER).with_suffix('')
image_count = len(list(data_dir.glob('*.jpg')))
print(image_count)

l = list(data_dir.glob('*'))
print(l[0])

Image.open(str(l[0]))

# %%
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,  # Scale pixel values between 0 and 1
    # rotation_range=50,

    # Other preprocessing options like rotation, flipping, etc.
) 

data_dir = paths.BW_IMG_FOLDER
batch_size = 128

image_generator = datagen.flow_from_directory(
    data_dir,
    color_mode="grayscale",
    target_size=(80, 60),  # Set your desired image dimensions
    batch_size=batch_size,
    class_mode='input',  # No class labels, unsupervised learning
    shuffle=True  # Shuffle the data
)


# %%
importlib.reload(ploters)
it = image_generator.next()
images = it[0]
print(images[1].shape)
ploters.plot_generated_images(images, 1, 5)



# %%

print(images.shape)
train_x_flatten=np.reshape(images,(images.shape[0],-1))
print(train_x_flatten.shape)

# %%
importlib.reload(v1)
vae, vae_encoder, vae_decoder = v1.getVAE(4800)
v1.plotVAE(vae)

# %%
kl_coefficient=1

#Information needed to compute the loss function
vae_input=vae.input
vae_output=vae.output
mu=vae.get_layer('mu').output
log_var=vae.get_layer('log_var').output

vae.add_loss(v1.vae_loss(vae_input,vae_output,mu,log_var,kl_coefficient,4800))
vae.compile(optimizer='adam')

# %%
epoch_count = 100
batch_size=100
patience=5

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

history = vae.fit(image_generator,epochs=epoch_count,callbacks=[early_stop])
# %%
epochs = 10
batch_size = 128

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    for batch_x in image_generator:  # Assuming the generator yields (batch_x, batch_y)
        loss = vae.train_on_batch(batch_x[0],batch_x[0])
        print(f"Batch loss: {loss}")
# %%
print(batch_x[0].shape)
# %%
