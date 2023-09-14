import numpy as np
from tensorflow import keras
from keras import layers
from keras import backend as K
import matplotlib.pyplot as plt
import random
import time
from sklearn.model_selection import train_test_split
from utils.gan_utils import chunks as chk
import utils.ploters as ploters
import utils.gan_utils as gu1


class cdcGan:
    def build_cdcgan(self, input_noise_dim, condition_dim):
        input_noise=layers.Input(shape=input_noise_dim, name='input_noise')
        input_condition=layers.Input(shape=condition_dim, name='input_condition')

        input_noise_reshaped=layers.Reshape((1,1,input_noise_dim))(input_noise)
        input_condition_reshaped=layers.Reshape((1,1,condition_dim))(input_condition)

        #Generator
        generator_input = layers.Concatenate(name='generator_input')([input_noise_reshaped, input_condition_reshaped])

        prev_layer=layers.Conv2DTranspose(256,7,strides=1,padding='valid')(generator_input)
        prev_layer=layers.BatchNormalization()(prev_layer)
        prev_layer=layers.LeakyReLU(alpha=0.2)(prev_layer)

        prev_layer=layers.Conv2DTranspose(128,5,strides=2,padding='same')(prev_layer)
        prev_layer=layers.BatchNormalization()(prev_layer)
        prev_layer=layers.LeakyReLU(alpha=0.2)(prev_layer)

        generator_output=layers.Conv2DTranspose(1,5,strides=2,padding='same',activation='tanh',name='generator_output')(prev_layer)

        generator = keras.Model([input_noise,input_condition], generator_output, name='generator')

        #Discriminator
        discriminator_input_sample = layers.Input(shape=(28,28,1), name='discriminator_input_sample')
  
        input_condition_dense=layers.Dense(784)(input_condition)
        discriminator_input_condition=layers.Reshape((28,28,1))(input_condition_dense)
  
        discriminator_input = layers.Concatenate(name='discriminator_input')([discriminator_input_sample, discriminator_input_condition])

        prev_layer=layers.Conv2D(128,5,strides=2,padding='same')(discriminator_input)
        prev_layer=layers.BatchNormalization()(prev_layer)
        prev_layer=layers.LeakyReLU(alpha=0.2)(prev_layer)

        prev_layer=layers.Conv2D(256,5,strides=2,padding='same')(prev_layer)
        prev_layer=layers.BatchNormalization()(prev_layer)
        prev_layer=layers.LeakyReLU(alpha=0.2)(prev_layer)

        prev_layer=layers.Conv2D(1,7,strides=1,padding='valid',activation='sigmoid')(prev_layer)
        discriminator_output=layers.Reshape((1,),name='discriminator_output')(prev_layer)

        discriminator = keras.Model([discriminator_input_sample,input_condition], discriminator_output, name='discriminator')

        #cDCGAN
        cdcgan = keras.Model(generator.input, discriminator([generator.output,input_condition]),name='cdcgan')
  
        return cdcgan,generator,discriminator