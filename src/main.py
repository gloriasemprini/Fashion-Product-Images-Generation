# %%
import tensorflow as tf
from tensorflow import keras
import importlib
import time
from keras.utils import to_categorical

import utils.paths as paths
import utils.callbacks as callbacks
import utils.ploters as ploters 

import vae_models.ccvae as ccvae

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

CLASSES = ["Watches"]


def labels_provider(l, n): 
   while len(l) > 0:
      poped = l[:n]
      l = l[n:]
      yield poped

# %%
importlib.reload(img_gen)
importlib.reload(paths)
importlib.reload(ploters)
importlib.reload(fid)
importlib.reload(preprocess)

# %% DF Generator
importlib.reload(img_gen)
importlib.reload(preprocess)

#parameters
BATCH_SIZE = 128
image_heigh = 80
image_weigh = 80
num_color_dimensions = 3 # 1 for greyscale or 3 for RGB
with_color_label = True # class label inlude article color

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


# %% VAE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
importlib.reload(ccvae)

latent_space_dimension = 64
internal_dense_layers = [1024, 512]
vae, vae_encoder, vae_decoder = ccvae.CCVAE().build_ccvae(
   image_shape, 
   internal_dense_layers, 
   latent_space_dimension,
   'LeakyReLU',
   one_hot_label_len)

vae.summary()
keras.utils.plot_model(vae, show_shapes=True, show_layer_names=True, expand_nested=True)

### Compilation
kl_coefficient=1
#Information needed to compute the loss function
vae_input=vae.input
vae_output=vae.output
mu=vae.get_layer('mu').output
log_var=vae.get_layer('log_var').output

vae.add_loss(ccvae.vae_loss(vae_input,vae_output,mu,log_var,kl_coefficient,num_pixels))
vae.compile(optimizer=tf.keras.optimizers.Adam(clipnorm=0.001)) # for debag  run_eagerly=True


loss_metrics=[]
val_loss_metrics=[]
fid_frequency_metrics = []




# %% =========================================== Manual training
importlib.reload(fid)
importlib.reload(img_gen)

epoch_count = 64
image_plot_frequency = 8
fid_frequency = 8 #

def batch_eleboration(model, generator, validation=False):
   n = 0
   loss_sum = 0
   for _ in range(len(generator)):
      batch_x, batch_y = next(generator)
      if(validation):
         loss = model.test_on_batch([batch_x, batch_y], batch_x)
      else:
         loss = model.train_on_batch([batch_x, batch_y], batch_x)
      n += len(batch_x)
      loss_sum += (loss*len(batch_x))
   return loss_sum / n


for e in range(1, epoch_count+1):
   start_time = time.time()

   loss = batch_eleboration(vae, train_provider)
   loss_metrics.append(loss)

   val_loss = batch_eleboration(vae, val_provider, validation=True)
   val_loss_metrics.append(val_loss)
   
   end_time = time.time()
   print('Epoch: {0} exec_time={1:.1f}s  loss={2:.3f} val_loss={3:.3f}'.format(e,end_time - start_time, loss, val_loss))

   if(e%image_plot_frequency == 0):
      ploters.plot_model_input_and_output(val_provider, vae) 
   if(is_fid_active and e%fid_frequency == 0):
      image_generator = img_gen.ConditionalImageGeneratorDecoder(vae_decoder, labels_provider(all_one_hot_labels, BATCH_SIZE))
      fid_frequency_metrics.append(fid.compute_fid(train_provider, image_generator, image_shape))

ploters.plot_model_input_and_output(val_provider, vae)

ploters.plot_losses_from_array(loss_metrics,val_loss_metrics)

if(is_fid_active):
   ploters.plot_fid(fid_frequency_metrics)



# %% Autogenerate new images
importlib.reload(ploters)
importlib.reload(img_gen)
if(with_color_label):
   ploters.plot_model_generated_colorfull_article_types(vae_decoder, len(CLASSES), one_hot_label_len, rows=2)
else:
   ploters.plot_model_generated_article_types(vae_decoder, one_hot_label_len, rows=1, cols=10)

if(latent_space_dimension == 2):
   ploters.plot_2d_latent_space(vae_decoder, image_shape)




# %%  ============= Automatic TRAINING==================== not work with label inputs
epoch_count = 2
patience=10

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
intermediateImages = callbacks.CustomFitCallback(val_provider, epochs_per_fit=10)

history = vae.fit(
   train_provider, 
   epochs=epoch_count,
   validation_data=val_provider,
   callbacks=[early_stop, intermediateImages])

#Print loss chart
ploters.plot_history(history)

# Trasform 5 random images from validation set
train_x = next(val_provider)

# get first 5 dataset images
watches = train_x[0][:5] if(type(train_x) is tuple) else train_x[:5]
ploters.plot_generated_images([watches], 1, 5)
generated_watches = vae.predict(watches)
ploters.plot_generated_images([generated_watches], 1, 5)