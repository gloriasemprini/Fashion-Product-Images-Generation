from keras.preprocessing.image import ImageDataGenerator
import utils.ploters as ploters 
import importlib

importlib.reload(ploters)

def createImageGenerator(data_dir, batch_size=2033):

    datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,  # Scale pixel values between 0 and 1
        rotation_range=15,
        validation_split=0.2
        # brightness_range=[0.5, 0.5]
    ) 

    train_data_generator = datagen.flow_from_directory(
        data_dir,
        color_mode="grayscale",
        target_size=(80, 60),  # Set your desired image dimensions
        batch_size=batch_size,
        class_mode='input',  # No class labels, unsupervised learning
        shuffle=True, # Shuffle the data
        subset='training'
    )

    validation_data_generator = datagen.flow_from_directory(
        data_dir,
        color_mode="grayscale",
        target_size=(80, 60),  # Set your desired image dimensions
        batch_size=batch_size,
        class_mode='input',  # No class labels, unsupervised learning
        shuffle=True,  # Shuffle the data
        subset='validation'
    )

    return train_data_generator, validation_data_generator

    
def plotGeneratedImages(generator):
  
    it = generator.next()
    images = it[0]
    print(images[1].shape)
    ploters.plot_generated_images([images], 1, 5)

    print("Images shape (numImages, high, width, numColors):")
    print(images.shape)
    # train_x_flatten=np.reshape(images,(images.shape[0],-1))
    # print(train_x_flatten.shape)