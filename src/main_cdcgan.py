# %% Importazioni
import matplotlib.pyplot as plt
import random
from scipy import linalg
from tensorflow import keras
import importlib

# import graphviz

import utils.paths as paths
import utils.ploters as ploters 
from utils.image_provider import labels_provider
import utils.image_provider as img_gen
import gan.gan as g
import gan.cdcgan_today as cdcg
import utils.gan_utils as g_ut
from keras.utils import to_categorical
import utils.df_preprocessing as preprocess

# %% Ricarica dei moduli
importlib.reload(img_gen)
importlib.reload(paths)
importlib.reload(ploters)
importlib.reload(g)
importlib.reload(g_ut)
importlib.reload(cdcg)

# %% Definizione delle classi
CLASSES = ["Sunglasses"]

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

# %% CDCGAN
importlib.reload(cdcg)
importlib.reload(g)
importlib.reload(g_ut)

input_noise_dim = 100
use_one_sided_labels = True

cdcgan, cdcgan_generator, cdcgan_discriminator = cdcg.cdcGan().build_cdcgan(input_noise_dim, one_hot_label_len, image_shape, num_pixels)

cdcgan_generator.summary()
cdcgan_discriminator.summary()
g_ut.plotcdcGAN(cdcgan)

# Ottimizzatori
optimizer_gen = keras.optimizers.Adam(learning_rate=0.0001)
optimizer_dis = keras.optimizers.Adam(learning_rate=0.0001)

cdcgan_discriminator.compile(loss='binary_crossentropy', optimizer=optimizer_dis)
cdcgan_discriminator.trainable = False
cdcgan.compile(loss='binary_crossentropy', optimizer=optimizer_gen)

# %% -------------------------------- Allenamento CDCGAN
epoch_count = 100
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

# %% FID (Commentato)
# La parte del FID è stata rimossa/commentata.
# 
# image_generator = img_gen.ConditionalGANImageGenerator(cdcgan_generator, labels_provider(all_one_hot_labels, BATCH_SIZE))
# fid.compute_fid(train_provider, image_generator, image_shape)
