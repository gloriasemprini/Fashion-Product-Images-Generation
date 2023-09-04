from tensorflow import keras
from keras import layers
import vae_models.vae as vae
from keras import backend as K

class ConvVae(vae.Vae):
    def build_vae(self, shape, input_count, neuron_count_per_hidden_layer,encoded_dim,hidden_activation,output_activation, num_classes=None):
        myshape = shape
        #Encoder
        encoder_input = layers.Input(shape=myshape, name='encoder_input')
        prev_layer = encoder_input
        channels = 16
        prev_layer = self.create_conv_block(prev_layer, channels, 4)
        for i in range(4):
            channels *= 2 
            prev_layer = self.create_downsampling_conv_block(prev_layer, channels)
        
        # prev_layer = self.create_conv_block(prev_layer, channels/2, 4)
        last_conv_shape = K.int_shape(prev_layer)

        prev_layer = layers.Flatten(name="Flatten")(prev_layer)
        
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
        

        n = last_conv_shape[1] * last_conv_shape[2] * last_conv_shape[3]
        prev_layer = layers.Dense(n, activation=hidden_activation)(prev_layer)
        prev_layer = layers.Reshape((last_conv_shape[1],last_conv_shape[2], last_conv_shape[3]))(prev_layer)
        channels = last_conv_shape[3]

        # prev_layer = self.create_conv_block(prev_layer, channels*2, 4)

        for i in range(4):
            channels //= 2 
            prev_layer = self.create_upsampling_conv_block(prev_layer, channels)
        prev_layer = layers.Conv2D(3, 4, padding="same", use_bias=False)(prev_layer)
        # prev_layer =layers.BatchNormalization()(prev_layer)
        decoder_output_layer = layers.Activation(output_activation, name='rec_image')(prev_layer)
        # prev_layer = layers.Conv2DTranspose(32, (3, 3), strides=1, padding="same", activation=hidden_activation)(prev_layer)
        # prev_layer = layers.Conv2DTranspose(16, (4, 4), strides=2, padding="same", activation=hidden_activation)(prev_layer)
  

        # decoder_output_layer=layers.Dense(input_count,activation=output_activation, name='decoder_output')(prev_layer)
        # decoder_output_layer2=layers.Reshape(myshape, name="Reshape")(decoder_output_layer)

        decoder = keras.Model(decoder_input, decoder_output_layer, name='decoder')

        #Sampling layer
        s = layers.Lambda(self.sampling, output_shape=(encoded_dim,), name='s')([mu, log_var])

        #VAE
        vae=keras.Model(encoder.input, decoder(s),name='vae')

        return vae,encoder,decoder
    
    def create_conv_block(self, prev_layer, channels, kernel_size=3, padding='same'):
        prev_layer = layers.Conv2D(channels, kernel_size, padding=padding, use_bias=False)(prev_layer)
        prev_layer = layers.BatchNormalization()(prev_layer)
        prev_layer = layers.LeakyReLU()(prev_layer) 
        return prev_layer

    def create_downsampling_conv_block(self, prev_layer, channels, kernel_size=4):
        prev_layer = layers.ZeroPadding2D()(prev_layer)
        prev_layer = layers.Conv2D(channels, kernel_size, strides=(2, 2), use_bias=False)(prev_layer)
        prev_layer = layers.BatchNormalization()(prev_layer)
        prev_layer = layers.LeakyReLU()(prev_layer) 
        return prev_layer

    def create_upsampling_conv_block(self, prev_layer, channels, kernel_size = 3):
        # prev_layer = layers.UpSampling2D()(prev_layer)
        prev_layer = layers.Conv2DTranspose(channels, kernel_size,strides=2, padding="same", activation="LeakyReLU")(prev_layer)
        # prev_layer = self.create_conv_block(prev_layer, channels, kernel_size = kernel_size)
        return prev_layer