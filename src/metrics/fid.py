from scipy import linalg
import tensorflow as tf
import numpy as np
import utils.image_provider as img_gen
import math 

# %% FID metric
def getInceptionModel(image_shape):
    inception_model = tf.keras.applications.InceptionV3(include_top=False, 
                                input_shape=image_shape,
                                weights="imagenet", 
                                pooling='avg')
    return inception_model

def compute_embeddings(dataloader, count, inception_model):
    image_embeddings = []
    it = iter(dataloader)
    for _ in range(count):
        images = next(it)
        if(type(images) is tuple):
            images, _ = images
        embeddings = inception_model.predict(images, verbose=0)
        image_embeddings.extend(embeddings)
    return np.array(image_embeddings)


def calculate_fid(real_embeddings, generated_embeddings):
    # calculate mean and covariance statistics
    mu1, sigma1 = real_embeddings.mean(axis=0), np.cov(real_embeddings, rowvar=False)
    mu2, sigma2 = generated_embeddings.mean(axis=0), np.cov(generated_embeddings,  rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = linalg.sqrtm(sigma1.dot(sigma2))

    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return round(math.sqrt(fid),2)

def compute_fid(real_img_generator, fake_img_generator, image_shape ):
    count = len(real_img_generator) - 1

    inception_model = getInceptionModel(image_shape)

    # compute embeddings for real images
    real_image_embeddings = compute_embeddings(real_img_generator, count, inception_model)

    # compute embeddings for generated images
    generated_image_embeddings= compute_embeddings(fake_img_generator, count, inception_model)

    if(len(generated_image_embeddings) > len(real_image_embeddings)):
        generated_image_embeddings = generated_image_embeddings[:len(real_image_embeddings)]

    fid = calculate_fid(real_image_embeddings, generated_image_embeddings)
    print("FID: " + str(fid))
    return fid