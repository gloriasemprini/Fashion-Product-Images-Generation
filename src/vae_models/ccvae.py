from tensorflow import keras
from keras import layers
from keras import backend as K

class CCVAE():
    def __init__(self):
       self.num_downsampling = 2

    def build_ccvae(self, shape, dense_neurons, encoded_dim, hidden_activation, label_input_len):
        #Input
        encoder_input = layers.Input(shape=shape, name='encoder_img_input')
        label_input = layers.Input(shape=(label_input_len,), name='encoder_label_input')
        encoder_label = layers.Dense(shape[0]*shape[1], name="encoder_label_size_augmentation")(label_input)
        encoder_label = layers.Reshape((shape[0], shape[1], 1), name="encoder_label_reshape")(encoder_label)
        concatedated_input = layers.Concatenate(name="input_concatenator")([encoder_input, encoder_label])

        #Encoder
        prev_layer, last_conv_shape = self.build_encoder(concatedated_input, hidden_activation, dense_neurons)
        
        mu = layers.Dense(encoded_dim, name='mu')(prev_layer)
        log_var = layers.Dense(encoded_dim, name='log_var')(prev_layer)

        encoder = keras.Model([encoder_input, label_input], [mu, log_var], name='encoder')

        #Decoder
        decoder_input = layers.Input(shape=(encoded_dim,), name='decoder_input')
        concatedated_input_dec = layers.Concatenate(name="decoder_concat_input")([decoder_input, label_input])

        prev_layer = self.build_decoder(concatedated_input_dec, hidden_activation, dense_neurons, last_conv_shape)
        decoder_output_layer = layers.Conv2D(shape[2], 4, padding="same", activation='sigmoid')(prev_layer)

        decoder = keras.Model([decoder_input, label_input], decoder_output_layer, name='decoder')

        #Sampling layer
        s = layers.Lambda(self.sampling, output_shape=(encoded_dim,), name='s')([mu, log_var])

        #VAE
        vae=keras.Model(encoder.input, decoder([s, label_input]),name='ccvae')

        return vae,encoder,decoder
    
    def build_encoder(self, input, h_activation, dense_neurons):

      prev_layer = input
      channels = 16
      prev_layer = self.create_conv_block(prev_layer, channels, h_activation, 4, norm=True)
      for i in range(self.num_downsampling):
          channels *= 2 
          prev_layer = self.create_downsampling_conv_block(prev_layer, channels, h_activation)
      
      channels = channels/8
      prev_layer = self.create_conv_block(prev_layer, channels, h_activation, 1, norm=True)
      channels *= 2 
      prev_layer = self.create_downsampling_conv_block(prev_layer, channels, h_activation)
      


      last_conv_shape = K.int_shape(prev_layer)

      prev_layer = layers.Flatten(name="Flatten")(prev_layer)
      
      for neuron_count in dense_neurons:
          prev_layer=layers.Dense(neuron_count,activation=h_activation)(prev_layer)
      
      return prev_layer, last_conv_shape
        

    def build_decoder(self, input, h_activation, dense_neurons, last_conv_shape):
        prev_layer=input
        for neuron_count in reversed(dense_neurons):
            prev_layer=layers.Dense(neuron_count,activation=h_activation)(prev_layer)
    
        n = last_conv_shape[1] * last_conv_shape[2] * last_conv_shape[3]
        prev_layer = layers.Dense(n, activation=h_activation)(prev_layer)
        prev_layer = layers.Reshape((last_conv_shape[1],last_conv_shape[2], last_conv_shape[3]))(prev_layer)
        channels = last_conv_shape[3]

        channels /= 2 
        prev_layer = self.create_upsampling_conv_block(prev_layer, channels, h_activation)
        channels = channels*8
        prev_layer = self.create_conv_block(prev_layer, channels,h_activation, 1, norm=True)


        for i in range(self.num_downsampling):
            channels //= 2 
            prev_layer = self.create_upsampling_conv_block(prev_layer, channels, h_activation)
        return prev_layer

       
    def create_conv_block(self, prev_layer, channels, activation, kernel_size=3, padding='same', norm=False):
      prev_layer = layers.Conv2D(channels, kernel_size, padding=padding, use_bias=True)(prev_layer)
      if(norm):
          prev_layer = layers.BatchNormalization()(prev_layer)
      prev_layer = layers.Activation(activation)(prev_layer) 
      return prev_layer

    def create_downsampling_conv_block(self, prev_layer, channels, activation, kernel_size=3):
        prev_layer = layers.ZeroPadding2D()(prev_layer)
        prev_layer = layers.Conv2D(channels, kernel_size, strides=2, use_bias=False)(prev_layer)
        prev_layer = layers.BatchNormalization()(prev_layer)
        prev_layer = layers.Activation(activation)(prev_layer)  
        return prev_layer

    def create_upsampling_conv_block(self, prev_layer, channels, activation, kernel_size = 3):
        prev_layer = layers.Conv2DTranspose(channels, kernel_size,strides=2, padding="same")(prev_layer)
        prev_layer = layers.BatchNormalization()(prev_layer)
        prev_layer = layers.Activation(activation)(prev_layer) 
        return prev_layer

    def sampling(self, args):
      mu, log_var = args
      batch_size = K.shape(mu)[0]
      dim = K.int_shape(mu)[1]
      epsilon = K.random_normal(shape=(batch_size, dim), mean=0., stddev=1.0)
      return K.exp(0.5 * log_var) * epsilon + mu
    

def vae_loss(vae_input,vae_ouput,mu,log_var,kl_coefficient, input_count):
  #Reconstruction loss
  vae_input = vae_input[0] if(type(vae_input) is list) else vae_input
  x = keras.layers.Reshape((input_count,))(vae_input)
  y = keras.layers.Reshape((input_count,))(vae_ouput)
  reconstruction_loss = keras.losses.mean_squared_error(x, y) * input_count
  # reconstruction_loss = keras.losses.mean_squared_error(vae_input,vae_ouput) * input_count

  #Regularization loss
  kl_loss = 0.5 * K.sum(K.square(mu) + K.exp(log_var) - log_var - 1, axis = -1)

  print(reconstruction_loss)
  print(kl_loss)
  #Combined loss
  return reconstruction_loss + kl_coefficient*kl_loss