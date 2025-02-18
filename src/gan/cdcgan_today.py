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
        # Generator
        kernel_ext = 10
        kernel_int = 4

        channels_arr = [256, 128, 64]
        channels_ext = 512
        stride = 2

        input_noise = layers.Input(shape=(input_noise_dim,), name='input_noise')  # Definito un input con dimensione (100,)
        input_condition = layers.Input(shape=(condition_dim,), name='input_condition')  # Etichetta condizionale

        input_noise_reshaped = layers.Reshape((1, 1, input_noise_dim))(input_noise)
        input_condition_reshaped = layers.Reshape((1, 1, condition_dim))(input_condition)

        # Concatenate noise and condition
        generator_input = layers.Concatenate(name='generator_input')([input_noise_reshaped, input_condition_reshaped])
        gen_prev_layer = generator_input

        gen_prev_layer = self.create_conv_2D_trans(gen_prev_layer, channels_ext, kernel_ext, 1, padding="valid", norm=True)

        for channels in channels_arr:
            gen_prev_layer = self.create_conv_2D_trans(gen_prev_layer, channels, kernel_int, stride, padding='same', norm=True)

        # Final layer for RGB images (3 channels)
        generator_output = layers.Conv2DTranspose(image_shape[2], kernel_int, 1, padding='same', activation='tanh', name='generator_output')(gen_prev_layer)

        generator = keras.Model([input_noise, input_condition], generator_output, name='generator')

        # Discriminator
        discriminator_input_sample = layers.Input(shape=image_shape, name='discriminator_input_sample')
        input_condition_dense = layers.Dense(image_shape[0] * image_shape[1])(input_condition)
        discriminator_input_condition = layers.Reshape((image_shape[0], image_shape[1], 1))(input_condition_dense)

        discriminator_input = layers.Concatenate(name='discriminator_input')([discriminator_input_sample, discriminator_input_condition])
        disc_prev_layer = discriminator_input
        disc_prev_layer = self.create_conv_2D(disc_prev_layer, 64, kernel_int, stride, padding='same', norm=True)

        for channels in [128, 256]:
            disc_prev_layer = self.create_conv_2D(disc_prev_layer, channels, kernel_int, stride, padding='same', norm=True)

        disc_prev_layer = self.create_conv_2D(disc_prev_layer, channels_ext, kernel_int, 1, padding='same', norm=True)
        disc_prev_layer = layers.Conv2D(1, kernel_ext, 1, padding='valid', activation='sigmoid')(disc_prev_layer)
        discriminator_output = layers.Reshape((1,), name='discriminator_output')(disc_prev_layer)

        discriminator = keras.Model([discriminator_input_sample, input_condition], discriminator_output, name='discriminator')

        # cDCGAN model
        cdcgan = keras.Model(generator.input, discriminator([generator.output, input_condition]), name='cdcgan')

        return cdcgan, generator, discriminator

    def create_conv_2D_trans(self, layer, channels, kernel, stride, padding, norm=True):
        layer = layers.Conv2DTranspose(channels, kernel, strides=stride, padding=padding)(layer)
        #if norm:
            #layer = layers.BatchNormalization()(layer)
        layer = layers.LeakyReLU(alpha=0.2)(layer)
        return layer
    
    def create_conv_2D(self, layer, channels, kernel, stride, padding, norm=True):
        layer = layers.Conv2D(channels, kernel, strides=stride, padding=padding)(layer)
        #if norm:
            #layer = layers.BatchNormalization()(layer)
        layer = layers.LeakyReLU(alpha=0.2)(layer)
        return layer
    
    def train_gan(self, gan, generator, discriminator, train_generator, train_data_count, input_noise_dim, epoch_count, batch_size,
              get_random_input_func, get_real_batch_func, get_fake_batch_func, concatenate_batches_func, condition_count=-1,
              use_one_sided_labels=False, plt_frq=None, plt_example_count=10, image_shape=(28, 28)):
        iteration_count = len(train_generator)
        print('Epochs: ', epoch_count)
        print('Batch size: ', batch_size)
        print('Iterations: ', iteration_count)
        train_data_count = batch_size * iteration_count
        print('Num images: ', train_data_count)
        print('')

        # Plot generated images before training
        if plt_frq is not None:
            print('Before training:')
            noise_to_plot = get_random_input_func(plt_example_count, input_noise_dim, condition_count)
            generated_output = generator.predict(noise_to_plot, verbose=0)
            generated_output = (generated_output * 255).astype(np.uint8)
            generated_images = generated_output
            ploters.plot_generated_images([generated_images], 1, plt_example_count, figsize=(15, 5))

        d_epoch_losses = []
        g_epoch_losses = []
        for e in range(1, epoch_count + 1):
            start_time = time.time()
            avg_d_loss = 0
            avg_g_loss = 0

            # Training loop
            for i in range(iteration_count):
                # 1. Create a batch with real images from the training set
                real_batch_x, real_batch_y = get_real_batch_func(train_generator, 0.9 if use_one_sided_labels else 1)

                current_batch_size = real_batch_x[0].shape[0]
                #print('current_batch_size: ', current_batch_size)
                # 2. Create noise vectors for the generator and generate the images from the noise
                generator_input = get_random_input_func(current_batch_size, input_noise_dim, condition_count)
                fake_batch_x, fake_batch_y = get_fake_batch_func(generator, current_batch_size, generator_input)

                # 3. Concatenate real and fake batches into a single batch
                discriminator_batch_x = concatenate_batches_func(real_batch_x, fake_batch_x)
                discriminator_batch_y = np.concatenate((real_batch_y, fake_batch_y))

                # 4. Train the discriminator
                for _ in range(2):  # Addestra il discriminatore più frequentemente
                    d_loss = discriminator.train_on_batch(discriminator_batch_x, discriminator_batch_y)
                
                # 5. Create noise vectors for the generator
                g_loss_sum = 0
                for _ in range(10): 
                    gan_batch_x = get_random_input_func(current_batch_size, input_noise_dim, condition_count)
                    #gan_batch_y = np.ones(current_batch_size)  # Flipped labels for training generator
                    gan_batch_y = np.random.uniform(0.8, 1.0, current_batch_size)
                    
                    # 6. Train the generator
                    g_loss = gan.train_on_batch(gan_batch_x, gan_batch_y)
                    g_loss_sum += g_loss
                    if g_loss < 2.0:
                        break
                g_loss = g_loss_sum / 10
                #print(f'd_loss={d_loss:.3f} g_loss={g_loss:.3f}')
                # 7. Average losses
                avg_d_loss += d_loss * current_batch_size
                avg_g_loss += g_loss * current_batch_size
                #print(f'avg_d_loss={avg_d_loss:.3f} avg_g_loss={avg_g_loss:.3f} current_batch_size={current_batch_size}')
            avg_d_loss /= train_data_count
            avg_g_loss /= train_data_count

            d_epoch_losses.append(avg_d_loss)
            g_epoch_losses.append(avg_g_loss)

            end_time = time.time()

            print(f'Epoch: {e} exec_time={end_time - start_time:.1f}s d_loss={avg_d_loss:.3f} g_loss={avg_g_loss:.3f}')

            # Update the plots
            if plt_frq is not None and e % plt_frq == 0:
                generated_output = generator.predict(noise_to_plot, verbose=0)
                generated_output = (generated_output + 1) * 127.5  # Riconversione da [-1,1] a [0,255]
                generated_output = np.clip(generated_output, 0, 255).astype(np.uint8)  # Assicura il range corretto
                generated_images = generated_output.reshape(plt_example_count, image_shape[0], image_shape[1], image_shape[2])
                ploters.plot_generated_images([generated_images], 1, plt_example_count, figsize=(15, 5))

        return d_epoch_losses, g_epoch_losses
    