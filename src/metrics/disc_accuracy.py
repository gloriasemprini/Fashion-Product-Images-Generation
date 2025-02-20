from keras.metrics import BinaryAccuracy
import numpy as np

# Crea un oggetto BinaryAccuracy
accuracy = BinaryAccuracy()

def calculate_discriminator_accuracy(real_images, fake_images, discriminator):
    # Classifica le immagini reali e false
    real_labels = np.ones((real_images.shape[0], 1))
    fake_labels = np.zeros((fake_images.shape[0], 1))
    
    real_preds = discriminator.predict(real_images)
    fake_preds = discriminator.predict(fake_images)
    
    # Calcola l'accuratezza per le immagini reali e false
    accuracy_real = accuracy.update_state(real_labels, real_preds)
    accuracy_fake = accuracy.update_state(fake_labels, fake_preds)
    
    return accuracy_real, accuracy_fake
