from tensorflow import keras
from keras import layers
import vae_models.convVAE as convVae
from keras import backend as K

class cConvVae(convVae.ConvVae):
    def build_vae(self, shape, input_count, neuron_count_per_hidden_layer,encoded_dim,hidden_activation,output_activation, num_classes=None):
        if(num_classes == None):
            raise Exception("Conditional Vae need the number of classes") 
        #Encoder
        encoder_input = layers.Input(shape=shape, name='encoder_input')
        label_input = layers.Input(shape=(num_classes,), name='encoder_label_input')
        encoder_label = layers.Dense(shape[0]*shape[1])(label_input)
        encoder_label = layers.Reshape((shape[0], shape[1], 1))(encoder_label)
        concatedated = layers.Concatenate(name="concatenator")([encoder_input, encoder_label])

        prev_layer = concatedated
        channels = 16
        prev_layer = self.create_conv_block(prev_layer, channels, hidden_activation, 4)
        num_downsampling = 4
        for i in range(num_downsampling):
            channels *= 2 
            prev_layer = self.create_downsampling_conv_block(prev_layer, channels, hidden_activation)
        
        channels = channels/2
        prev_layer = self.create_conv_block(prev_layer, channels,hidden_activation, 4)

        last_conv_shape = K.int_shape(prev_layer)

        prev_layer = layers.Flatten(name="Flatten")(prev_layer)
        
        for neuron_count in neuron_count_per_hidden_layer:
            hidden_layer=layers.Dense(neuron_count,activation=hidden_activation)(prev_layer)
            prev_layer=hidden_layer

        mu = layers.Dense(encoded_dim, name='mu')(prev_layer)
        log_var = layers.Dense(encoded_dim, name='log_var')(prev_layer)

        encoder = keras.Model([encoder_input, label_input], [mu, log_var], name='encoder')

        #Decoder
        decoder_input = layers.Input(shape=(encoded_dim,), name='decoder_input')
        concatedated = layers.Concatenate(name="decoder_concat_input")([decoder_input, label_input])

        prev_layer=concatedated
        for neuron_count in reversed(neuron_count_per_hidden_layer):
            hidden_layer=layers.Dense(neuron_count,activation=hidden_activation)(prev_layer)
            prev_layer=hidden_layer
        

        n = last_conv_shape[1] * last_conv_shape[2] * last_conv_shape[3]
        prev_layer = layers.Dense(n, activation=hidden_activation)(prev_layer)
        prev_layer = layers.Reshape((last_conv_shape[1],last_conv_shape[2], last_conv_shape[3]))(prev_layer)
        channels = last_conv_shape[3]

        channels = channels*2
        prev_layer = self.create_conv_block(prev_layer, channels,hidden_activation, 4)


        for i in range(num_downsampling):
            channels //= 2 
            prev_layer = self.create_upsampling_conv_block(prev_layer, channels, hidden_activation)

        decoder_output_layer = layers.Conv2D(shape[2], 4, padding="same", use_bias=False, activation=output_activation)(prev_layer)


        decoder = keras.Model([decoder_input, label_input], decoder_output_layer, name='decoder')

        #Sampling layer
        s = layers.Lambda(self.sampling, output_shape=(encoded_dim,), name='s')([mu, log_var])

        #VAE
        vae=keras.Model(encoder.input, decoder([s, label_input]),name='ccvae')

        return vae,encoder,decoder
