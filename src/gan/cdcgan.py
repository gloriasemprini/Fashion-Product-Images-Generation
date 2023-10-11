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
    def build_cdcgan(self, input_noise_dim, condition_dim, image_shape, n_pixels):
        #Generator
        #provare con 5 o 6
        kernel_ext = 16
        #provare con anche 2
        kernel_int = 8
        
        input_noise=layers.Input(shape=input_noise_dim, name='input_noise')
        input_condition=layers.Input(shape=condition_dim, name='input_condition')

        input_noise_reshaped=layers.Reshape((1,1,input_noise_dim))(input_noise)
        input_condition_reshaped=layers.Reshape((1,1,condition_dim))(input_condition)

        channels_arr = [256,128,64]
        channels_ext = 512
        stride = 2

        #Generator
        generator_input = layers.Concatenate(name='generator_input')([input_noise_reshaped, input_condition_reshaped])
        gen_prev_layer = generator_input
        # model, layer, channels, kernel, stride, ccpadding, norm=True
        gen_prev_layer = self.create_conv_2D_trans(gen_prev_layer, generator_input, channels_ext, kernel_ext, 1, padding="valid", norm=True)
        print('shape:', gen_prev_layer.shape)
        gen_prev_layer = self.create_conv_2D_trans(gen_prev_layer, gen_prev_layer, 256, kernel_int, stride, padding="same", norm=True)
        print('shape:', gen_prev_layer.shape)
        gen_prev_layer = self.create_conv_2D_trans(gen_prev_layer, gen_prev_layer, 128, kernel_int, stride, padding="same", norm=True)
        print('shape:', gen_prev_layer.shape)
        gen_prev_layer = self.create_conv_2D_trans(gen_prev_layer, gen_prev_layer, 64, kernel_int, 1, padding="same", norm=True)
        print('shape:', gen_prev_layer.shape)


        #for channels in channels_arr:
        #    if channels != 256:
        #        prev_layer = self.create_conv_2D_trans(prev_layer, prev_layer, channels, kernel_int, stride, padding="same", norm=True)
        #    else:
        #        prev_layer = self.create_conv_2D_trans(prev_layer, prev_layer, channels, kernel_int, stride, padding="same", norm=True)

        generator_output=layers.Conv2DTranspose(image_shape[2],kernel_int,1,padding='same',activation='tanh',name='generator_output')(gen_prev_layer)

        generator = keras.Model([input_noise,input_condition], generator_output, name='generator')

        #Discriminator
        discriminator_input_sample = layers.Input(shape=image_shape, name='discriminator_input_sample')
  
        input_condition_dense=layers.Dense(n_pixels)(input_condition)
        discriminator_input_condition=layers.Reshape(image_shape)(input_condition_dense)
  
        discriminator_input = layers.Concatenate(name='discriminator_input')([discriminator_input_sample, discriminator_input_condition])
        disc_prev_layer = discriminator_input
        disc_prev_layer = self.create_conv_2D(disc_prev_layer, discriminator_input, 64, kernel_int, stride, padding='same', norm=True)
        print('shape:', disc_prev_layer.shape)
        disc_prev_layer = self.create_conv_2D(disc_prev_layer, disc_prev_layer, 128, kernel_int, stride, padding='same', norm=True)
        print('shape:', disc_prev_layer.shape)
        #disc_prev_layer = self.create_conv_2D(disc_prev_layer, disc_prev_layer, 256, kernel_int, stride, padding='same', norm=True)
        #print('shape:', disc_prev_layer.shape)

        #for channels in reversed(channels_arr):
        #    if channels != 64:
        #        prev_layer = self.create_conv_2D(prev_layer, prev_layer, channels, kernel_int, stride, padding='same', norm=True)

        disc_prev_layer = self.create_conv_2D(disc_prev_layer, disc_prev_layer, channels_ext, kernel_int, 1, padding='same', norm=True)
        print('shape:', disc_prev_layer.shape)
        disc_prev_layer=layers.Conv2D(1,kernel_ext,1,padding='valid',activation='sigmoid')(disc_prev_layer)
        print('shape:', disc_prev_layer.shape)
        discriminator_output=layers.Reshape((1,),name='discriminator_output')(disc_prev_layer)

        discriminator = keras.Model([discriminator_input_sample,input_condition], discriminator_output, name='discriminator')

        #cDCGAN
        cdcgan = keras.Model(generator.input, discriminator([generator.output,input_condition]),name='cdcgan')
  
        return cdcgan,generator,discriminator
    
    def create_conv_2D_trans(self, model, layer, channels, kernel, stride, padding, norm=True):
        model=layers.Conv2DTranspose(channels, kernel, strides=stride, padding=padding)(layer)
        model=layers.BatchNormalization()(model)
        model=layers.LeakyReLU(alpha=0.2)(model)
        return model
    
    def create_conv_2D(self, model, layer, channels, kernel, stride, padding, norm=True):
        model=layers.Conv2D(channels,kernel,stride,padding)(layer)
        model=layers.BatchNormalization()(model)
        model=layers.LeakyReLU(alpha=0.2)(model)
        return model