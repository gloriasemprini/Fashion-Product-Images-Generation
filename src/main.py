# %%
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
import importlib
import time

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
num_color_dimensions = 3 # 1 for greyscale or 3 for RGB
BATCH_SIZE = 128
imageSize = (image_heigh, image_weigh)
with_color_label = True # class label inlude article color

if(with_color_label and num_color_dimensions != 3): # error check
   raise Exception("Illegal state: color label can be used only with RGB images")

class_mode = "multi_output" if(with_color_label) else "categorical"
train_generator, validation_generator  = img_gen.create_image_generator_df(
    paths.IMG_FOLDER,
    CLASSES,
    class_mode=class_mode,
    image_size=imageSize,
    batch_size=BATCH_SIZE,
    rgb=(num_color_dimensions==3),
)
one_hot_label_len = train_generator.num_classes if(with_color_label) else len(train_generator.class_indices)


img_gen.plotGeneratedImages(train_generator)


# %% VAE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
importlib.reload(v1)
importlib.reload(flat_vae)
importlib.reload(conv_vae)
importlib.reload(cconv_vae)


NUM_PIXELS = image_heigh * image_weigh * num_color_dimensions
image_shape = (image_heigh, image_weigh, num_color_dimensions)
latent_space_dimension = 32
vae, vae_encoder, vae_decoder = cconv_vae.cConvVae().build_vae(
   image_shape, 
   NUM_PIXELS, 
   [2048, 1024], 
   latent_space_dimension,
   'LeakyReLU',
   'sigmoid',
   one_hot_label_len)

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

epoch_count = 32
image_plot_frequency = 1

def batch_eleboration(model, generator, validation=False):
   n = 0
   avg_loss = 0
   for i in range(len(generator)):
      batch_x, batch_y = next(generator)
      if(validation):
         loss = model.test_on_batch([batch_x, batch_y], batch_x)
      else:
         loss = model.train_on_batch([batch_x, batch_y], batch_x)
      n += len(batch_x)
      avg_loss += (loss*len(batch_x))
   avg_loss = avg_loss / n
   return avg_loss


for e in range(1, epoch_count+1):
   avg_loss = 0
   avg_val_loss = 0
   n = 0

   start_time = time.time()
   avg_loss = batch_eleboration(vae, train_generator)
   loss_metrics.append(avg_loss)

   avg_val_loss = batch_eleboration(vae, validation_generator, validation=True)
   val_loss_metrics.append(avg_val_loss)
   
   end_time = time.time()
   print('Epoch: {0} exec_time={1:.1f}s  loss={2:.3f} val_loss={3:.3f}'.format(e,end_time - start_time, avg_loss, avg_val_loss))

   if(e%image_plot_frequency == 0):
      ploters.plot_model_input_and_output(validation_generator, vae) 

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
importlib.reload(ploters)
importlib.reload(img_gen)
if(with_color_label):
   ploters.plot_model_generated_colorfull_article_types(vae_decoder, len(CLASSES), one_hot_label_len, rows=1)
else:
   ploters.plot_model_generated_article_types(vae_decoder, one_hot_label_len, rows=1, cols=10)

if(latent_space_dimension == 2):
   ploters.plot_2d_latent_space(vae_decoder, image_shape)

# %% Work only with RGB images
importlib.reload(fid)
image_generator = img_gen.ConditionalImageGeneratorDecoder(vae_decoder, BATCH_SIZE, label=one_hot)
fid.getFid(train_generator, image_generator, image_shape)

