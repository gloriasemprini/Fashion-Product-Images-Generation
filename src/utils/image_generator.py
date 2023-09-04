from keras.preprocessing.image import ImageDataGenerator
import utils.ploters as ploters 
import importlib
import numpy as np
import random
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

importlib.reload(ploters)

def createImageGenerator(data_dir, batch_size=64, imageSize = (80,60), rgb=False, class_mode=None):
    color_mode =  "rgb" if (rgb) else "grayscale" 

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
        class_mode=class_mode,#'input', 'categorical' 
        shuffle=True, # Shuffle the data
        subset='training'
    )

    validation_data_generator = datagen.flow_from_directory(
        data_dir,
        color_mode=color_mode,
        target_size=imageSize,
        batch_size=batch_size,
        class_mode=class_mode,#'input', 'categorical' 
        shuffle=True,  # Shuffle the data
        subset='validation'
    )

    return train_data_generator, validation_data_generator

def create_image_generator_df(df, data_dir, batch_size=64, imageSize = (80,60), rgb=False, class_mode=None):
    color_mode =  "rgb" if (rgb) else "grayscale" 
    
    articleType_encoder = LabelEncoder()
    color_encoder = LabelEncoder()
    articleType_encoder.fit(df["articleType"].unique())
    color_encoder.fit(df["baseColour"].unique())

    datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,  # Scale pixel values between 0 and 1
        validation_split=0.1,
    ) 

    train_data_generator = datagen.flow_from_dataframe(
        df,
        data_dir,
        x_col="id",
        y_col=["articleType", "baseColour"],
        color_mode=color_mode,
        target_size=imageSize,
        batch_size=batch_size,
        class_mode=class_mode,#'input', 'categorical' 
        shuffle=True, 
        subset='training'
    )

    validation_data_generator = datagen.flow_from_dataframe(
        df,
        data_dir,
        x_col="id",
        y_col=["articleType", "baseColour"],
        color_mode=color_mode,
        target_size=imageSize,
        batch_size=batch_size,
        class_mode=class_mode,#'input', 'categorical' 
        shuffle=True,
        subset='validation'
    )
    train_data_generator = MultilabelImageDataGenerator(train_data_generator, articleType_encoder, color_encoder)
    validation_data_generator = MultilabelImageDataGenerator(validation_data_generator, articleType_encoder, color_encoder)
    
    return train_data_generator, validation_data_generator
def plotGeneratedImages(generator):
  
    it = next(generator)
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
    
class MultilabelImageDataGenerator:
    def __init__(self, generator, articleType_encoder, color_encoder):
        self.generator = generator
        self.articleType_encoder = articleType_encoder
        self.color_encoder = color_encoder
        self.articl_n_classes =  len(articleType_encoder.classes_)
        self.color_n_classes =  len(color_encoder.classes_)
        self.num_classes = self.articl_n_classes + self.color_n_classes 
        self.class_indicies = articleType_encoder.classes_

    def __iter__(self):
        return self
    
    def __next__(self):
        x, y = next(self.generator)
        articleType_one_hot = to_categorical(self.articleType_encoder.transform(y[0]), num_classes=self.articl_n_classes)
        color_one_hot = to_categorical(self.color_encoder.transform(y[1]), num_classes=self.color_n_classes)
        concatenated = []
        for i in range(len(articleType_one_hot)):
            concatenated.append(articleType_one_hot[i].tolist() + color_one_hot[i].tolist())
        
        return x, np.array(concatenated, dtype=np.float32)
    
    def __len__(self):
        return len(self.generator)

    