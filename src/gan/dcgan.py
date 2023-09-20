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
    def build_dcgan(self, input_noise_dim, image_shape):
        #Generator
        #provare con 5 o 6
        kernel_ext = 8
        #provare con anche 2
        kernel_int = 4
        generator = keras.Sequential(name='generator')

        stride = 2

        generator.add(layers.Input(shape=input_noise_dim, name='generator_input'))
        generator.add(layers.Reshape((1,1,input_noise_dim)))
        
        channels_arr = [128,64]
        channels_ext = 256

        generator = self.create_conv_2D_trans(generator, channels_ext, kernel_ext, stride, padding="valid", norm=True)

        for channels in channels_arr:
            generator = self.create_conv_2D_trans(generator, channels, kernel_int, stride, padding="same")

        generator.add(layers.Conv2DTranspose(image_shape[2],kernel_int,stride,padding='same',activation='tanh',name='generator_output'))

        #Discriminator
        discriminator = keras.Sequential(name='discriminator')
        
        discriminator.add(layers.Input(shape=image_shape,name='discriminator_input'))
        
        for channels in reversed(channels_arr):
            discriminator = self.create_conv_2D(discriminator, channels, kernel_int, stride, padding="same")

        discriminator = self.create_conv_2D(discriminator, channels_ext, kernel_int, stride, padding="same", norm=True)

        discriminator.add(layers.Conv2D(1,kernel_ext,stride,padding='valid',activation='sigmoid'))
        discriminator.add(layers.Reshape((1,),name='discriminator_output'))

        #DCGAN
        dcgan = keras.Model(generator.input, discriminator(generator.output),name='dcgan')

        return dcgan,generator,discriminator
    

    def create_conv_2D_trans(self, model, channels, kernel, stride, padding, norm=True):
        model.add(layers.Conv2DTranspose(channels,kernel,strides=stride,padding=padding))
        if (norm):
            model.add(layers.BatchNormalization())
        model.add(layers.ReLU())
        return model
    
    def create_conv_2D(self, model, channels, kernel, stride, padding, norm=True):
        model.add(layers.Conv2D(channels,kernel,strides=stride,padding=padding))
        if (norm):
            model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))
        return model