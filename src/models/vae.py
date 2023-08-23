import numpy as np
from tensorflow import keras
from keras import layers
from keras import backend as K

class Vae:
  def sampling(self, args):
      mu, log_var = args
      batch_size = K.shape(mu)[0]
      dim = K.int_shape(mu)[1]
      epsilon = K.random_normal(shape=(batch_size, dim), mean=0., stddev=1.0)
      return K.exp(0.5 * log_var) * epsilon + mu

  def build_vae(self, shape, input_count, neuron_count_per_hidden_layer,encoded_dim,hidden_activation,output_activation):
      #Encoder
      encoder_input = layers.Input(shape=input_count, name='encoder_input')
      encoder_input2 = layers.Reshape((80,60,1), input_shape=(4800,), name='encoder_input_reshaped')(encoder_input)
      
      # encoder_input = layers.Input(shape=(80,60,1), name='encoder_input')
      prev_layer=encoder_input2
      prev_layer = layers.Conv2D(16, (4, 4), strides=(2, 2), padding="same", activation=hidden_activation)(prev_layer)
      prev_layer = layers.MaxPool2D()(prev_layer)
      prev_layer = layers.Conv2D(32, (3,3),  padding="same", activation=hidden_activation)(prev_layer)
      prev_layer = layers.MaxPool2D()(prev_layer)
      # prev_layer = layers.Conv2D(16)(prev_layer)

      prev_layer = layers.Flatten()(prev_layer)
      for neuron_count in neuron_count_per_hidden_layer:
        hidden_layer=layers.Dense(neuron_count,activation=hidden_activation)(prev_layer)
        prev_layer=hidden_layer
      
      mu = layers.Dense(encoded_dim, name='mu')(prev_layer)
      log_var = layers.Dense(encoded_dim, name='log_var')(prev_layer)
      
      encoder = keras.Model(encoder_input, [mu, log_var], name='encoder')

      #Decoder
      decoder_input = layers.Input(shape=(encoded_dim,), name='decoder_input')

      prev_layer=decoder_input
      for neuron_count in reversed(neuron_count_per_hidden_layer):
        hidden_layer=layers.Dense(neuron_count,activation=hidden_activation)(prev_layer)
        prev_layer=hidden_layer
      
      prev_layer = layers.Dense(1680,activation=hidden_activation)(prev_layer)
      prev_layer = layers.Reshape((10, 7, 24))(prev_layer)
      prev_layer = layers.Conv2DTranspose(24, (3, 3), strides=1, padding="same", activation=hidden_activation)(prev_layer)
      prev_layer = layers.Conv2DTranspose(16, (4, 4), strides=2, padding="same", activation=hidden_activation)(prev_layer)
      # prev_layer = layers.Conv2D(32, (40,30))(prev_layer)
      # prev_layer = layers.Conv2D(1, (80, 60), padding="same")(prev_layer)
      prev_layer = layers.Flatten()(prev_layer)

      decoder_output_layer=layers.Dense(input_count,activation=output_activation, name='decoder_output')(prev_layer)

      # prev_layer = layers.Conv2D(32, (20,15))(prev_layer)
      # prev_layer = layers.Dense(4800,activation=hidden_activation)(prev_layer)
      # decoder_output_layer=layers.Reshape((80,60,1), name='decoder_output')(prev_layer)

      decoder = keras.Model(decoder_input, decoder_output_layer, name='decoder')

      #Sampling layer
      s = layers.Lambda(self.sampling, output_shape=(encoded_dim,), name='s')([mu, log_var])

      #VAE
      vae=keras.Model(encoder.input, decoder(s),name='vae')
      
      return vae,encoder,decoder

  def getVAE(self, shape, count):
    vae, vae_encoder, vae_decoder=self.build_vae(shape, count, [1024, 512, 128], 32,'relu','sigmoid')
    vae.summary()
    return vae, vae_encoder, vae_decoder

def plotVAE(vae):
  keras.utils.plot_model(vae, show_shapes=True, show_layer_names=True, expand_nested=True)

def vae_loss(vae_input,vae_ouput,mu,log_var,kl_coefficient, input_count):
  #Reconstruction loss
  # reconstruction_loss = keras.losses.mean_squared_error(vae_input,vae_ouput) * input_count
  x = keras.layers.Reshape((input_count,))(vae_input)
  y = keras.layers.Reshape((input_count,))(vae_ouput)
  reconstruction_loss = keras.losses.mean_squared_error(x, y) * input_count

  #Regularization loss
  kl_loss = 0.5 * K.sum(K.square(mu) + K.exp(log_var) - log_var - 1, axis = -1)

  #Combined loss
  return reconstruction_loss + kl_coefficient*kl_loss
