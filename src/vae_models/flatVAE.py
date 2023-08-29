from tensorflow import keras
from keras import layers
import vae_models.vae as vae

class FlatVAE(vae.Vae):
    def build_vae(self, shape, input_count, neuron_count_per_hidden_layer,encoded_dim,hidden_activation,output_activation):
        #Encoder
        encoder_input = layers.Input(shape=input_count, name='encoder_input')

        prev_layer=encoder_input

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

        decoder = keras.Model(decoder_input, decoder_output_layer, name='decoder')

        #Sampling layer
        s = layers.Lambda(self.sampling, output_shape=(encoded_dim,), name='s')([mu, log_var])

        #VAE
        vae=keras.Model(encoder.input, decoder(s),name='vae')

        return vae,encoder,decoder