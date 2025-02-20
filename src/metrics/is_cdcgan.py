import numpy as np
import tensorflow as tf
from keras.applications.inception_v3 import InceptionV3, preprocess_input

# Carica il modello InceptionV3
inception_model = InceptionV3(weights='imagenet', include_top=True, pooling='avg')

def calculate_inception_score(generated_images, num_samples=100):
    generated_images = generated_images[:num_samples]

    if isinstance(generated_images, list):
        generated_images = np.array(generated_images)

    # Step intermedio: prima porta a 160x160, poi a 299x299
    images_resized = np.array([
        tf.image.resize(tf.image.resize(img, (160, 160)), (299, 299)).numpy()
        for img in generated_images
    ])

    images = preprocess_input(images_resized)

    # Predizioni
    preds = inception_model.predict(images)

    # Evita log(0)
    epsilon = 1e-8
    preds = np.maximum(preds, epsilon)
    preds /= np.sum(preds, axis=1, keepdims=True) + epsilon

    # KL-divergence
    kl_divergence = np.sum(preds * (np.log(preds) - np.log(np.mean(preds, axis=0))), axis=1)
    inception_score = np.exp(np.mean(kl_divergence))

    return inception_score
