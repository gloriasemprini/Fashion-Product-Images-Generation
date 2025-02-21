# %% Importazioni
import matplotlib.pyplot as plt
import random
from scipy import linalg
from tensorflow import keras
import importlib
import numpy as np
import tensorflow as tf
import metrics.fid_cdcgan as fid
import metrics.is_cdcgan as is_cdc

# import graphviz
import utils.paths as paths
import utils.ploters as ploters 
from utils.image_provider import labels_provider
import utils.image_provider as img_gen
import gan.gan as g
import gan.cdcgan_metrics as cdcg
import utils.gan_utils as g_ut
from keras.utils import to_categorical
import utils.df_preprocessing as preprocess

# set seed
seed_value = 42  # Puoi scegliere un altro numero, l'importante è che sia fisso
random.seed(seed_value)  # Per la randomizzazione standard di Python
np.random.seed(seed_value)  # Per NumPy
tf.random.set_seed(seed_value)  # Per TensorFlow

# %% Ricarica dei moduli
importlib.reload(img_gen)
importlib.reload(paths)
importlib.reload(ploters)
importlib.reload(g)
importlib.reload(g_ut)
importlib.reload(cdcg)
importlib.reload(is_cdc)
importlib.reload(fid)

# %% Definizione delle classi
CLASSES = ["Belts"]
#CLASSES = ["Watches", "Handbags", "Sunglasses", "Belts", "Backpacks", "Sarees", "Deodorant", "Nail Polish", "Ties", "Flip Flops", "Formal Shoes"]

import os

img_folder = paths.IMG_FOLDER
all_images = os.listdir(img_folder)

for img in all_images:
    img_path = os.path.join(img_folder, img)
    if not os.path.exists(img_path):
        print(f"Immagine mancante: {img_path}")


# %% DF Generator
importlib.reload(img_gen)
importlib.reload(preprocess)

# Parametri
BATCH_SIZE = 64
image_heigh = 80  # Impostato a 80x80 come richiesto
image_weigh = 80  # Impostato a 80x80 come richiesto
num_color_dimensions = 3  # 1 per scala di grigi o 3 per RGB
with_color_label = True  # Etichetta di classe include il colore dell'articolo

# Parametri calcolati
image_size = (image_heigh, image_weigh)
image_shape = (image_heigh, image_weigh, num_color_dimensions)
num_pixels = image_heigh * image_weigh * num_color_dimensions
rgb_on = (num_color_dimensions == 3)

if with_color_label and not rgb_on:  # Verifica per errore
    raise Exception("Stato illegale: l'etichetta di colore può essere utilizzata solo con immagini RGB")

class_mode = "multi_output" if with_color_label else "categorical"
train_provider, _ = img_gen.create_data_provider_df(
    paths.IMG_FOLDER,
    CLASSES,
    class_mode=class_mode,
    image_size=image_size,
    batch_size=BATCH_SIZE,
    rgb=rgb_on,
    validation_split=0.001,
    tanh_rescale=True
)

one_hot_label_len = train_provider.num_classes if with_color_label else len(train_provider.class_indices)
if isinstance(train_provider, img_gen.MultiLabelImageDataGenerator):
    all_one_hot_labels = train_provider.labels
else:
    all_one_hot_labels = to_categorical(train_provider.labels)

ploters.plot_provided_images(train_provider)

# %% GPU
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
#tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

# %% CDCGAN
importlib.reload(cdcg)
importlib.reload(g)
importlib.reload(g_ut)

import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

input_noise_dim = 100
use_one_sided_labels = True

cdcgan, cdcgan_generator, cdcgan_discriminator = cdcg.cdcGan().build_cdcgan(input_noise_dim, one_hot_label_len, image_shape, num_pixels)

cdcgan_generator.summary()
cdcgan_discriminator.summary()
g_ut.plotcdcGAN(cdcgan)

# Ottimizzatori
optimizer_gen = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
optimizer_dis = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

cdcgan_discriminator.compile(loss='binary_crossentropy', optimizer=optimizer_dis)
cdcgan_discriminator.trainable = False
cdcgan.compile(loss='binary_crossentropy', optimizer=optimizer_gen)

# %% Caricamento pesi (se esistono)
try:
    cdcgan_generator.load_weights("generator_weights.h5")
    cdcgan_discriminator.load_weights("discriminator_weights.h5")
    print("Pesi caricati con successo.")
except:
    print("Nessun peso trovato. Il modello verrà allenato da zero.")

# %% -------------------------------- Allenamento CDCGAN
epoch_count = 52
d_epoch_losses, g_epoch_losses = cdcg.cdcGan().train_gan(
    cdcgan,
    cdcgan_generator,
    cdcgan_discriminator,
    train_provider,
    2000,
    input_noise_dim,
    epoch_count,
    BATCH_SIZE,
    g_ut.get_cgan_random_input,
    g_ut.get_cgan_real_batch,
    g_ut.get_cgan_fake_batch,
    g_ut.concatenate_cgan_batches,
    condition_count=one_hot_label_len,
    use_one_sided_labels=use_one_sided_labels,
    plt_frq=2,
    plt_example_count=10,
    image_shape=image_shape
)

ploters.plot_gan_losses(d_epoch_losses, g_epoch_losses)

# %% Salvataggio dei pesi
cdcgan_generator.save_weights("generator_weights_sunglasses.h5")
cdcgan_discriminator.save_weights("discriminator_weights_sunglasses.h5")
print("Pesi salvati con successo.")

# %% Generazione delle immagini
importlib.reload(ploters)
importlib.reload(img_gen)

if with_color_label:
    ploters.plot_model_generated_colorfull_article_types(
        cdcgan_generator, 
        len(CLASSES), 
        one_hot_label_len, 
        rows=1,
        imgProducer=img_gen.ConditionalGANImageGenerator
    )
else:
    ploters.plot_model_generated_article_types(
        cdcgan_generator, 
        one_hot_label_len, 
        rows=1, 
        cols=10,
        imgProducer=img_gen.ConditionalGANImageGenerator
    )

# %% FID
importlib.reload(fid)
# Esegui il calcolo del FID
fid_value = fid.compute_fid(train_provider, cdcgan_generator, BATCH_SIZE, image_size)

print(f"FID: {fid_value}")

# %%
