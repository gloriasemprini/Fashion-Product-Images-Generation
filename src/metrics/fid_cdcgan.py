import numpy as np
import tensorflow as tf
from keras.applications.inception_v3 import InceptionV3
from scipy.linalg import sqrtm
import os

# Funzione per caricare il modello InceptionV3
def load_inception_model():
    model = InceptionV3(include_top=False, weights='imagenet', pooling='avg')
    return model

# Funzione per calcolare le feature delle immagini tramite InceptionV3
def get_inception_features(imgs, model, batch_size=64):
    features = []
    for i in range(0, len(imgs), batch_size):
        batch = imgs[i:i+batch_size]
        batch = tf.convert_to_tensor(batch, dtype=tf.float32)
        batch = tf.keras.applications.inception_v3.preprocess_input(batch)
        batch_features = model(batch).numpy()
        features.append(batch_features)
    return np.concatenate(features, axis=0)

# Funzione per calcolare la distanza Fréchet
def calculate_fid(real_features, generated_features):
    mu_real, sigma_real = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu_gen, sigma_gen = generated_features.mean(axis=0), np.cov(generated_features, rowvar=False)
    
    # Calcola la distanza Fréchet
    diff = mu_real - mu_gen
    covmean = sqrtm(sigma_real.dot(sigma_gen))
    fid = np.sum(diff**2) + np.trace(sigma_real + sigma_gen - 2.0 * covmean)
    return fid

# Funzione per generare le immagini e calcolare il FID
def compute_fid(real_data_provider, generator, batch_size, image_size, num_samples=1000):
    # Ottieni le immagini reali e generate
    real_data, generated_images = get_images_for_fid(real_data_provider, generator, batch_size, image_size, num_samples)
    
    # Carica il modello InceptionV3
    inception_model = load_inception_model()

    # Calcola le feature delle immagini reali
    real_features = get_inception_features(real_data, inception_model, batch_size)
    
    # Calcola le feature delle immagini generate
    generated_features = get_inception_features(generated_images, inception_model, batch_size)

    # Calcola e restituisce il FID
    fid_value = calculate_fid(real_features, generated_features)
    fid_value = np.real(fid_value)
    return fid_value


# Funzione per ottenere immagini reali e generate per il calcolo del FID
def get_images_for_fid(real_data_provider, generator, batch_size, image_size, num_samples=None):
    # Se non viene specificato num_samples, usa il numero di immagini nel dataset
    if num_samples is None:
        num_samples = len(real_data_provider)  # Calcola il numero di immagini disponibili

    # Assicurati che num_samples sia un multiplo del batch_size
    num_samples = (num_samples // batch_size) * batch_size

    real_images = []
    real_labels = []  # Salva anche le etichette reali
    for i in range(num_samples // batch_size):
        real_batch, labels = next(real_data_provider)  # Ottieni immagini e etichette
        real_images.append(real_batch)
        real_labels.append(labels)
    
    real_images = np.concatenate(real_images, axis=0)
    real_labels = np.concatenate(real_labels, axis=0)

    # Verifica che il numero di immagini reali sia uguale a num_samples
    assert real_images.shape[0] == num_samples, f"Numero di immagini reali non corrisponde a num_samples: {real_images.shape[0]} != {num_samples}"

    # Genera rumore per il generatore
    noise = np.random.normal(0, 1, (num_samples, 100))  # 100 è la dimensione del rumore
    
    # Le etichette sono già raccolte da `real_data_provider`
    generated_images = generator.predict([noise, real_labels])  # Passa sia il rumore che le etichette
    
    # Verifica che il numero di immagini generate sia uguale a num_samples
    assert generated_images.shape[0] == num_samples, f"Numero di immagini generate non corrisponde a num_samples: {generated_images.shape[0]} != {num_samples}"
    
    return real_images, generated_images



