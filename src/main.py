# %%
import math
import numpy as np
# import matplotlib.pyplot as plt
import random
# from scipy import linalg
import tensorflow as tf
from tensorflow import keras
# from numba import cuda
# from tqdm import tqdm
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
import utils.df_preprocessing as preprocess

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


### 
# "Casual Shoes" #2846
# "Flats" #500
# "Heels" #1323



 
CLASSES = ["Watches", "Handbags", "Sunglasses", "Belts", "Flip Flops"]

# %%
importlib.reload(v1)
importlib.reload(img_gen)
importlib.reload(paths)
importlib.reload(ploters)
importlib.reload(fid)
importlib.reload(preprocess)

# %% DF Generator
importlib.reload(img_gen)
importlib.reload(preprocess)


image_heigh = 64
image_weigh = 64
NUM_COLORS = 3
BATCH_SIZE = 128
imageSize = (image_heigh, image_weigh)
with_color_label = True


class_mode = "multi_output" if(with_color_label) else "categorical"
train_generator, validation_generator  = img_gen.create_image_generator_df(
    paths.IMG_FOLDER,
    CLASSES,
    imageSize=imageSize,
    batch_size=BATCH_SIZE,
    rgb=(NUM_COLORS==3),
    class_mode=class_mode
)
num_classes = train_generator.num_classes if(with_color_label) else len(train_generator.class_indices)


img_gen.plotGeneratedImages(train_generator)


# %% VAE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
importlib.reload(v1)
importlib.reload(flat_vae)
importlib.reload(conv_vae)
importlib.reload(cconv_vae)


NUM_PIXELS = image_heigh * image_weigh * NUM_COLORS
image_shape = (image_heigh, image_weigh, NUM_COLORS)
latent_space_dimension = 64
vae, vae_encoder, vae_decoder = cconv_vae.cConvVae().build_vae(
   image_shape, 
   NUM_PIXELS, 
   [2048, 1024], 
   latent_space_dimension,
   'LeakyReLU',
   'sigmoid',
   num_classes)

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
vae.compile(optimizer=tf.keras.optimizers.Adam(clipnorm=0.001),run_eagerly=True) # for debag  run_eagerly=True


loss_metrics=[]
val_loss_metrics=[]

# val_generator.shuffle = False
# val_generator.index_array = None


# %% =========================================== Manual training

epoch_count = 2
image_plot_frequency = 2

def batch_eleboration(model, generator, validation=False):
   n = 0
   avg_loss = 0
   type = "validation" if validation else 'train'
   for i in range(len(generator)):
      batch_x, batch_y = next(generator)
      current_batch_size = len(batch_x)
      if(validation):
         loss = model.test_on_batch([batch_x, batch_y], batch_x)
      else:
         loss = model.train_on_batch([batch_x, batch_y], batch_x)
      if(i%2 == 0):
         print(".", end="")
      n += current_batch_size
      avg_loss += (loss*current_batch_size)
   avg_loss = avg_loss / n
   print(" loss of",type, avg_loss)
   return avg_loss

for e in range(1, epoch_count+1):
   avg_loss = 0
   avg_val_loss = 0
   n = 0

   avg_loss = batch_eleboration(vae, train_generator)
   loss_metrics.append(avg_loss)

   avg_val_loss = batch_eleboration(vae, validation_generator, validation=True)
   val_loss_metrics.append(avg_val_loss)

   if(e%image_plot_frequency == 0):
      print("current epoch is ",e)
      ploters.plot_model_input_and_output(validation_generator, vae) 
   # if(e%25):
   #    train_generator.shuffle = False
   #    train_generator.index_array = None

ploters.plot_model_input_and_output(validation_generator, vae)

ploters.plot_losses_from_array(loss_metrics,val_loss_metrics)
# %%  ============= Automatic TRAINING==================== not work with label inputs
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
importlib.reload(img_gen)
num_images = 5
images_in_cols = 5
rows = math.ceil(num_images/images_in_cols)

# decoderGen = img_gen.ImageGeneratorDecoder(vae_decoder, images_in_cols)
for i in range(len(preprocess.CLASSES), train_generator.num_classes):
   one_hot = np.zeros(train_generator.num_classes, dtype=float)
   one_hot[0] = 1
   one_hot[i] = 0.9
   decoderGen = img_gen.ConditionalImageGeneratorDecoder(vae_decoder, images_in_cols,label=one_hot)
   iterator = iter(decoderGen)

   generated_images=[]
   for row in range(rows):
      generated_images.append(next(iterator))      

   ploters.plot_generated_images(generated_images,rows,images_in_cols, figsize=(10, 5))

if(latent_space_dimension == 2):
   ploters.plot_2d_latent_space(vae_decoder, image_shape)


# %%
importlib.reload(img_gen)
num_images = len(preprocess.CLASSES)
images_in_cols = len(preprocess.CLASSES)
rows = math.ceil(num_images/images_in_cols)

# decoderGen = img_gen.ImageGeneratorDecoder(vae_decoder, images_in_cols)
arr = []
for clas in range(images_in_cols):
   generated_images=[]
   for color  in range(len(preprocess.CLASSES), train_generator.num_classes):
      one_hot = np.zeros(train_generator.num_classes, dtype=float)
      one_hot[clas] = 1
      one_hot[color] = 1
      decoderGen = img_gen.ConditionalImageGeneratorDecoder(vae_decoder, 1,label=one_hot)
      iterator = iter(decoderGen)
      generated_images.append(next(iterator)[0])    

   ploters.plot_generated_images([generated_images],rows,images_in_cols, figsize=(10, 5))
# %% Work only with RGB images
importlib.reload(fid)
image_generator = img_gen.ConditionalImageGeneratorDecoder(vae_decoder, BATCH_SIZE, label=one_hot)
fid.getFid(train_generator, image_generator, image_shape)

# %%
