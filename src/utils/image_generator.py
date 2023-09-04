from keras.preprocessing.image import ImageDataGenerator
import utils.ploters as ploters 
import importlib
import numpy as np
import random

importlib.reload(ploters)

def createImageGenerator(data_dir, batch_size=64, imageSize = (80,60), rgb=False, class_mode=None):
    color_mode = "grayscale" 
    if (rgb):
        color_mode = "rgb"

    datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,  # Scale pixel values between 0 and 1
        # rotation_range=10,
        validation_split=0.1,
        # width_shift_range=0.1,
        # height_shift_range=0.1,
        # brightness_range=[0.5, 0.5]
    ) 


    train_data_generator = datagen.flow_from_directory(
        data_dir,
        color_mode=color_mode,
        target_size=imageSize,
        batch_size=batch_size,
        class_mode=class_mode,#'input', 'categorical' # No class labels, unsupervised learning
        shuffle=True, # Shuffle the data
        subset='training'
    )

    validation_data_generator = datagen.flow_from_directory(
        data_dir,
        color_mode=color_mode,
        target_size=imageSize,
        batch_size=batch_size,
        class_mode=class_mode,#'input', 'categorical' # No class labels, unsupervised learning
        shuffle=True,  # Shuffle the data
        subset='validation'
    )

    return train_data_generator, validation_data_generator

    
def plotGeneratedImages(generator):
  
    it = generator.next()
    if(type(it) is tuple):
       images, labels = it
    else: 
        images = it #it[0]

 
    print("An image shape: ", images[1].shape)
    ploters.plot_generated_images([images], 1, 5)

    print("Images shape (numImages, high, width, numColors):")
    print(images.shape)
    # train_x_flatten=np.reshape(images,(images.shape[0],-1))
    # print(train_x_flatten.shape)


class ImageGeneratorDecoder:
    def __init__(self, model, batch_size):
        self.model = model
        self.batch_size = batch_size
        self.encoder_input_size = model.layers[0].input_shape[0][1]

    def __iter__(self):
        return self
    
    def __next__(self):
        inputs = []
        for k in range(self.batch_size):
            random_sample = []
            for i in range(self.encoder_input_size):
                random_sample.append(random.normalvariate(0,1))
            inputs.append(random_sample)
        generated_images = self.model.predict(np.array(inputs),verbose=0)
        
        return generated_images
    

    

class ConditionalImageGeneratorDecoder:
    def __init__(self, model, batch_size, label):
        self.model = model
        self.batch_size = batch_size
        self.encoder_input_size = model.layers[0].input_shape[0][1]
        self.labels = [label for _ in range(batch_size)]

    def __iter__(self):
        return self
    
    def __next__(self):
        inputs = []
        for k in range(self.batch_size):
            random_sample = []
            for i in range(self.encoder_input_size):
                random_sample.append(random.normalvariate(0,1))
            inputs.append(random_sample)
        generated_images = self.model.predict([np.array(inputs), np.array(self.labels)],verbose=0)
        
        return generated_images