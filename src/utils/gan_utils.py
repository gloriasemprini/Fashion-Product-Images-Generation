import numpy as np
import matplotlib.pyplot as plt
import random
import time
from tensorflow import keras


def chunks(list, n):
    for i in range(0, len(list), n):
        yield list[i:i + n]

def plotGAN(gan):
    keras.utils.plot_model(gan, show_shapes=True, show_layer_names=True, expand_nested=True)

def plotcGAN(cgan):
    keras.utils.plot_model(cgan,show_shapes=True, show_layer_names=True,expand_nested=True)

def plotdcGAN(dcgan):
    keras.utils.plot_model(dcgan,show_shapes=True, show_layer_names=True,expand_nested=True)

def plotcdcGAN(cdcgan):
    keras.utils.plot_model(cdcgan,show_shapes=True, show_layer_names=True,expand_nested=True)

#GAN
def get_random_batch_indices(data_count,batch_size):
    list_indices=list(range(0,data_count))
    random.shuffle(list_indices)
    return list(chunks(list_indices, batch_size))

def get_gan_real_batch(dataset_x,label):
    train_x, _ = next(dataset_x)
    #batch_x = dataset_x[batch_indices]
    #batch_y=np.full(len(batch_indices),label)
    n_elem_batch = train_x.shape[0]
    batch_y=np.full(n_elem_batch,label)
    return train_x,batch_y
  
def get_gan_random_input(batch_size,noise_dim,*_):
    return np.random.normal(0, 1, size=(batch_size, noise_dim))

def get_gan_fake_batch(generator,batch_size,generator_input):
    batch_x = generator.predict(generator_input,verbose=0)
    batch_y=np.zeros(batch_size)
    return batch_x,batch_y
  
def get_gan_random_input(batch_size,noise_dim,*_):
    return np.random.normal(0, 1, size=(batch_size, noise_dim))

def get_gan_fake_batch(generator,batch_size,generator_input):
    batch_x = generator.predict(generator_input,verbose=0)
    batch_y=np.zeros(batch_size)
    return batch_x,batch_y
  
def concatenate_gan_batches(real_batch_x,fake_batch_x):
    return np.concatenate((real_batch_x, fake_batch_x))

#cGAN
def get_cgan_real_batch(dataset,batch_indices,label):
  dataset_input=dataset[0]
  dataset_condition_info=dataset[1]
  batch_x =[dataset_input[batch_indices],dataset_condition_info[batch_indices]]
  batch_y=np.full(len(batch_indices),label)

  return batch_x,batch_y

def get_cgan_random_input(batch_size,noise_dim,condition_count):
  noise=np.random.normal(0, 1, size=(batch_size, noise_dim))
  condition_info= to_categorical(np.random.randint(0, condition_count, size=batch_size),condition_count)

  return [noise,condition_info]

def get_cgan_fake_batch(generator,batch_size,generator_input):
  batch_x = [generator.predict(generator_input,verbose=0),generator_input[1]]
  batch_y=np.zeros(batch_size)

  return batch_x,batch_y

def concatenate_cgan_batches(real_batch_x,fake_batch_x):
  batch_input = np.concatenate((real_batch_x[0], fake_batch_x[0]))
  batch_condition_info =np.concatenate((real_batch_x[1], fake_batch_x[1]))

  return [batch_input,batch_condition_info]