from tensorflow import keras
from keras import layers
import models.vae as vae

class ConvVae(vae.Vae):
    def build_vae(self, shape, input_count, neuron_count_per_hidden_layer,encoded_dim,hidden_activation,output_activation):
        myshape = shape
        #Encoder
        encoder_input = layers.Input(shape=myshape, name='encoder_input')
        
        prev_layer = layers.Flatten(name="Flatten")(encoder_input)

      

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

        decoder_output_layer=layers.Dense(input_count,activation=output_activation, name='decoder_output')(prev_layer)
        decoder_output_layer2=layers.Reshape(myshape, name="Reshape")(decoder_output_layer)

        decoder = keras.Model(decoder_input, decoder_output_layer2, name='decoder')

        #Sampling layer
        s = layers.Lambda(self.sampling, output_shape=(encoded_dim,), name='s')([mu, log_var])

        #VAE
        vae=keras.Model(encoder.input, decoder(s),name='vae')

        return vae,encoder,decoder