from keras.preprocessing.image import ImageDataGenerator
import utils.ploters as ploters 
import importlib
import numpy as np
import random
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import utils.df_preprocessing as preprocess

importlib.reload(ploters)

def createImageGenerator(
        data_dir, 
        batch_size=64, 
        imageSize = (80,60), 
        rgb=False, 
        class_mode=None):
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

def create_image_generator_df(
        data_dir, 
        classes,
        class_mode,
        batch_size=64, 
        image_size = (80,60), 
        rgb=False):
    """Create an producer of images

    Args:
        data_dir (string): path to the directory with images
        classes (list of strings): list with article types that should be uploaded 
        class_mode (str): "categorical" if one hot encoded label should contain only articleType.
            "multi_output" if one hot encoded label should contain articleType and baseColour.
        batch_size (int, optional): Number of images uploaded at each iteration. Defaults to 64.
        image_size (tuple, optional): image size. Defaults to (80,60).
        rgb (bool, optional): true if images should be with colors. Defaults to False.

    Returns:
        MultiLabelImageDataGenerator: for "multi_output"  class_mode
        DataFrameIterator: for "categorical"and others class_mode
    """
    color_mode =  "rgb" if (rgb) else "grayscale" 
    y = ["articleType", "baseColour"] if(class_mode=="multi_output") else "articleType"

    df = preprocess.filter_articles(preprocess.get_clean_DF(), classes=classes)
    def append_ext(id): return id+".jpg"
    df['id'] = df['id'].apply(append_ext)
    
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
        y_col=y,
        color_mode=color_mode,
        target_size=image_size,
        batch_size=batch_size,
        class_mode=class_mode,#'input', 'categorical' 
        shuffle=True, 
        subset='training'
    )

    validation_data_generator = datagen.flow_from_dataframe(
        df,
        data_dir,
        x_col="id",
        y_col=y,
        color_mode=color_mode,
        target_size=image_size,
        batch_size=batch_size,
        class_mode=class_mode,#'input', 'categorical' 
        shuffle=True,
        subset='validation'
    )
    if(class_mode=="multi_output"):
        train_data_generator = MultiLabelImageDataGenerator(train_data_generator, articleType_encoder, color_encoder)
        validation_data_generator = MultiLabelImageDataGenerator(validation_data_generator, articleType_encoder, color_encoder)
    
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
    

    

# class ConditionalImageGeneratorDecoder:
#     def __init__(self, model, batch_size, labels):
#         self.model = model
#         self.batch_size = batch_size
#         self.encoder_input_size = model.layers[0].input_shape[0][1]
#         if(len(labels) == 1):
#             self.labels = [labels[0] for _ in range(batch_size)]
#         else:
#             if(len(labels) != batch_size):
#                 raise Exception("batch size must be equal to labels size")
#             self.labels = labels

#     def __iter__(self):
#         return self
    
#     def __next__(self):
#         inputs = []
#         for k in range(self.batch_size):
#             random_sample = []
#             for i in range(self.encoder_input_size):
#                 random_sample.append(random.normalvariate(0,0.6))
#             inputs.append(random_sample)
#         generated_images = self.model.predict([np.array(inputs), np.array(self.labels)],verbose=0)
        
#         return generated_images
    
class ConditionalImageGeneratorDecoder:
    def __init__(self, model,label_provider):
        self.model = model
        self.encoder_input_size = model.layers[0].input_shape[0][1]
        self.label_provider = label_provider

    def __iter__(self):
        return self
    
    def __next__(self):
        inputs = []
        labels = next(self.label_provider)
        for k in range(len(labels)):
            random_sample = []
            for i in range(self.encoder_input_size):
                random_sample.append(random.normalvariate(0,0.6))
            inputs.append(random_sample)
        generated_images = self.model.predict([np.array(inputs), np.array(labels)],verbose=0)
        
        return generated_images
    
class MultiLabelImageDataGenerator:
    def __init__(self, generator, articleType_encoder, color_encoder):
        self.generator = generator
        self.articleType_encoder = articleType_encoder
        self.color_encoder = color_encoder
        self.articl_n_classes =  len(articleType_encoder.classes_)
        self.color_n_classes =  len(color_encoder.classes_)
        self.num_classes = self.articl_n_classes + self.color_n_classes 
        self.class_indicies = articleType_encoder.classes_

        all_artcle = to_categorical(articleType_encoder.transform(generator.labels[0]))
        all_colors = to_categorical(color_encoder.transform(generator.labels[1]))

        concatenated = []
        for i in range(len(all_artcle)):
            concatenated.append(all_artcle[i].tolist() + all_colors[i].tolist())
        
        self.labels = np.array(concatenated, dtype=np.float32)

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
    

    