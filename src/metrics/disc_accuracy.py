from keras.metrics import BinaryAccuracy
import numpy as np

def calculate_discriminator_accuracy(real_images, real_labels, fake_images, fake_labels, discriminator):
    # Creiamo due array di etichette corrette: 1 per reali, 0 per fake
    real_targets = np.ones((real_images.shape[0], 1))
    fake_targets = np.zeros((fake_images.shape[0], 1))

    # Passiamo sia immagini che etichette al discriminatore
    real_preds = discriminator.predict([real_images, real_labels])
    fake_preds = discriminator.predict([fake_images, fake_labels])

    # Convertiamo le probabilità in valori binari (0 o 1)
    real_preds = (real_preds > 0.5).astype(np.float32)
    fake_preds = (fake_preds > 0.5).astype(np.float32)

    # Calcoliamo l'accuratezza separata per reali e fake
    accuracy_real = BinaryAccuracy()(real_targets, real_preds).numpy()
    accuracy_fake = BinaryAccuracy()(fake_targets, fake_preds).numpy()

    return accuracy_real, accuracy_fake
