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

class dcGan:
    def build_dcgan(self, input_noise_dim):
        #Generator
        layer_e = 8
        layer_i = 4
        strides_ge = 2
        strides_gi = 2
        strides_de = 2
        strides_di = 2
        generator = keras.Sequential(name='generator')

        generator.add(layers.Input(shape=input_noise_dim, name='generator_input'))
        generator.add(layers.Reshape((1,1,input_noise_dim)))

        """generator.add(layers.Conv2DTranspose(1024,layer_e,strides=1,padding='valid'))
        generator.add(layers.BatchNormalization(scale=True))
        generator.add(layers.LeakyReLU(alpha=0.2))"""

        generator.add(layers.Conv2DTranspose(4096,layer_i,strides=1,padding='valid'))
        #generator.add(layers.BatchNormalization(scale=True))
        generator.add(layers.BatchNormalization())
        generator.add(layers.LeakyReLU(alpha=0.2))
        
        generator.add(layers.Conv2DTranspose(1024,layer_i,strides=2,padding='same'))
        #generator.add(layers.BatchNormalization(scale=True))
        generator.add(layers.BatchNormalization())
        generator.add(layers.LeakyReLU(alpha=0.2))

        generator.add(layers.Conv2DTranspose(256,layer_i,strides=2,padding='same'))
        #generator.add(layers.BatchNormalization(scale=True))
        generator.add(layers.BatchNormalization())
        generator.add(layers.LeakyReLU(alpha=0.2))

        generator.add(layers.Conv2DTranspose(64,layer_i,strides=2,padding='same'))
        #generator.add(layers.BatchNormalization(scale=True))
        generator.add(layers.BatchNormalization())
        generator.add(layers.LeakyReLU(alpha=0.2))

        generator.add(layers.Conv2DTranspose(1,layer_i,strides=2,padding='same',activation='tanh',name='generator_output'))

        #Discriminator
        discriminator = keras.Sequential(name='discriminator')
        
        discriminator.add(layers.Input(shape=(64,64,1),name='discriminator_input'))
        
        discriminator.add(layers.Conv2D(64,layer_i,strides=2,padding='same'))
        #discriminator.add(layers.BatchNormalization(scale=True))
        discriminator.add(layers.BatchNormalization())
        discriminator.add(layers.LeakyReLU(alpha=0.2))

        discriminator.add(layers.Conv2D(256,layer_i,strides=2,padding='same'))
        #discriminator.add(layers.BatchNormalization(scale=True))
        discriminator.add(layers.BatchNormalization())
        discriminator.add(layers.LeakyReLU(alpha=0.2))

        discriminator.add(layers.Conv2D(1024,layer_i,strides=2,padding='same'))
        #discriminator.add(layers.BatchNormalization(scale=True))
        discriminator.add(layers.BatchNormalization())
        discriminator.add(layers.LeakyReLU(alpha=0.2))

        discriminator.add(layers.Conv2D(4096,layer_i,strides=1,padding='same'))
        #discriminator.add(layers.BatchNormalization(scale=True))
        discriminator.add(layers.BatchNormalization())
        discriminator.add(layers.LeakyReLU(alpha=0.2))

        """discriminator.add(layers.Conv2D(1024,layer_i,strides=1,padding='same'))
        discriminator.add(layers.BatchNormalization(scale=True))
        discriminator.add(layers.LeakyReLU(alpha=0.2))"""

        discriminator.add(layers.Conv2D(1,layer_e,strides=1,padding='valid',activation='sigmoid'))
        discriminator.add(layers.Reshape((1,),name='discriminator_output'))

        #DCGAN
        dcgan = keras.Model(generator.input, discriminator(generator.output),name='dcgan')

        return dcgan,generator,discriminator