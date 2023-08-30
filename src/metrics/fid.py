from scipy import linalg
import tensorflow as tf
import numpy as np
import utils.image_generator as img_gen

# %% FID metric
def getInceptionModel(image_shape):
    inception_model = tf.keras.applications.InceptionV3(include_top=False, 
                                input_shape=image_shape,
                                weights="imagenet", 
                                pooling='avg')
    return inception_model

def compute_embeddings(dataloader, count, inception_model):
    image_embeddings = []
    for _ in range(count):
        images = next(iter(dataloader))
        embeddings = inception_model.predict(images)
        image_embeddings.extend(embeddings)
    return np.array(image_embeddings)


def getFid(real_img_generator, fake_img_generator, image_shape ):
    count = 50 # math.ceil(10000/BATCH_SIZE)

    inception_model = getInceptionModel(image_shape)

    # compute embeddings for real images
    real_image_embeddings = compute_embeddings(real_img_generator, count, inception_model)

    # compute embeddings for generated images
    generated_image_embeddings = compute_embeddings(iter(fake_img_generator), count, inception_model)


    print("Real embedding shape: " + str(real_image_embeddings.shape))
    print("Generated embedding shape: " + str(generated_image_embeddings.shape))

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
            return fid


    fid = calculate_fid(real_image_embeddings, generated_image_embeddings)

    print("FID: " + str(fid))
    return fid