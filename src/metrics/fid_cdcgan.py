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
def compute_fid(real_data, generated_images, image_size, batch_size=64):
    # Carica il modello InceptionV3
    inception_model = load_inception_model()

    # Calcola le feature delle immagini reali
    real_features = get_inception_features(real_data, inception_model, batch_size)
    
    # Calcola le feature delle immagini generate
    generated_features = get_inception_features(generated_images, inception_model, batch_size)

    # Calcola e restituisce il FID
    fid_value = calculate_fid(real_features, generated_features)
    return fid_value

# Funzione per ottenere immagini reali e generate per il calcolo del FID
def get_images_for_fid(real_data_provider, generator, batch_size, image_size, num_samples=1000):
    real_images = []
    for i in range(num_samples // batch_size):
        real_batch, _ = next(real_data_provider)
        real_images.append(real_batch)
    
    real_images = np.concatenate(real_images, axis=0)
    
    noise = np.random.normal(0, 1, (num_samples, 100))
    generated_images = generator.predict(noise)
    
    return real_images, generated_images
