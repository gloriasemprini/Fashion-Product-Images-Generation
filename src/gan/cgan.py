import numpy as np
from tensorflow import keras
from keras import layers
from keras import backend as K
import matplotlib.pyplot as plt
import random
import time
from sklearn.model_selection import train_test_split
from tensorflow import keras
from utils.gan_utils import chunks as chk
import utils.ploters as ploters
import utils.gan_utils as gu1


class cGan:
  def build_cgan(self,input_noise_dim,neuron_count_per_hidden_layer,image_shape,n_pixel,hidden_activation,generator_output_activation, condition_dim):
    input_noise=layers.Input(shape=input_noise_dim, name='input_noise')
    input_condition=layers.Input(shape=condition_dim, name='input_condition')
    
    #Generator
    generator_input = layers.Concatenate(name='generator_input')([input_noise, input_condition])

    prev_layer=generator_input
    for neuron_count in neuron_count_per_hidden_layer:
      hidden_layer=layers.Dense(neuron_count,activation=hidden_activation)(prev_layer)
      prev_layer=hidden_layer

    generator_output = layers.Dense(n_pixel, activation=generator_output_activation,name='generator_output')(prev_layer)
    generator = keras.Model([input_noise,input_condition], generator_output, name='generator')

    #Discriminator
    discriminator_input_sample = layers.Input(shape=image_shape, name='discriminator_input_sample')
    discriminator_input = layers.Concatenate(name='discriminator_input')([discriminator_input_sample, input_condition])

    prev_layer=discriminator_input
    for neuron_count in reversed(neuron_count_per_hidden_layer):
      hidden_layer=layers.Dense(neuron_count,activation=hidden_activation)(prev_layer)
      prev_layer=hidden_layer

    discriminator_output = layers.Dense(1, activation='sigmoid',name='discriminator_output')(prev_layer)
    
    discriminator = keras.Model([discriminator_input_sample,input_condition], discriminator_output, name='discriminator')

    #cGAN
    cgan = keras.Model(generator.input, discriminator([generator.output,input_condition]),name='cgan')
  
    return cgan,generator,discriminator