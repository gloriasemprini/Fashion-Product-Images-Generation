import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image
import random
import utils.image_generator as img_gen
import utils.paths as paths


def plotRandomImg(df, num=15, path=paths.IMG_FOLDER):
  """ plot random 15 images from dataframe

  Args:
      df (dataframe): pandas dataframe
      num (int, optional): number of images. Defaults to 15.
      path (string, optional): path to the folder with images. Defaults to paths.IMG_FOLDER.
  """
  ids = df['id']
  selected_image_ids = random.sample(ids.tolist(), num)
  plotImagesById(selected_image_ids, path)

def plotImagesById(ids, folder=paths.IMG_FOLDER):
    """Plot images by their id

    Args:
        ids (list): is a list of image id, the number of id must be multiple of 5
    """
    rows = int(len(ids)/5)
    fig, axes = plt.subplots(rows, 5, figsize=(15, rows*3))

    for i, ax in enumerate(axes.flatten()):
        image_id = ids[i]
        image_path = paths.getImagePath(image_id, folder)  
        img = Image.open(image_path)
        ax.imshow(img, cmap='Greys_r')
        ax.set_title(image_id)
        ax.axis('off')

    plt.tight_layout()
    plt.show()





# Contains functions that allow plot data 

### Images
def plot_generated_images(generated_images, nrows, ncols,no_space_between_plots=False, figsize=(15, 15)):
  _, axs = plt.subplots(nrows, ncols, figsize=figsize,squeeze=False)

  for i in range(nrows):
    for j in range(ncols):
      axs[i,j].axis('off')
      axs[i,j].imshow(generated_images[i][j], cmap='gray')

  if no_space_between_plots:
    plt.subplots_adjust(wspace=0,hspace=0)

  plt.show()

# Used for showing autoencoder input and its corresponding output
def plot_model_input_and_output(generator, model, num=6):
   # Trasform 5 random images from validation set
   val_x, val_y = next(generator)
   if (len(val_x) < num):
      val_x, val_y = next(generator) # redo 

   # get first 5 dataset images
   real_imgs = val_x[:num] 
   labels = val_y[:num]
   plot_generated_images([real_imgs], 1, num)

   generated_imgs = model.predict([real_imgs,labels], verbose=0)
   plot_generated_images([generated_imgs], 1, num)

def plot_2d_latent_space(decoder, image_shape):
  n = 12 # number of images per row and column
  limit=3 # random values are sampled from the range [-limit,+limit]

  grid_x = np.linspace(-limit,limit, n) 
  grid_y = np.linspace(limit,-limit, n)

  generated_images=[]
  for i, yi in enumerate(grid_y):
    single_row_generated_images=[]
    for j, xi in enumerate(grid_x):
      random_sample = np.array([[ xi, yi]])
      decoded_x = decoder.predict(random_sample,verbose=0)
      single_row_generated_images.append(decoded_x[0])
    generated_images.append(single_row_generated_images)      

  plot_generated_images(generated_images,n,n,True)

### Plot generation
def plot_model_generated_article_types(model, one_hot_len, rows=1, cols=5):
  for i in range(one_hot_len):
    one_hot = np.zeros(one_hot_len, dtype=float)
    one_hot[i] = 1
    decoderGen = img_gen.ConditionalImageGeneratorDecoder(model, cols,label=[one_hot])
    iterator = iter(decoderGen)

    generated_images=[]
    for _ in range(rows):
        generated_images.append(next(iterator))      

    plot_generated_images(generated_images,rows,cols)

def plot_model_generated_colorfull_article_types(model, num_classes, one_hot_len, rows=1):
  num_colors = one_hot_len - num_classes
  for clas in range(num_classes):
    one_hots = []
    for color  in range(num_classes, one_hot_len):
        one_hot = np.zeros(one_hot_len, dtype=float)
        one_hot[clas] = 1
        one_hot[color] = 1
        one_hots.append(one_hot)
    decoderGen = img_gen.ConditionalImageGeneratorDecoder(model, batch_size=num_colors, labels=one_hots)

    for row in range(rows):
      plot_generated_images([next(iter(decoderGen))],1,num_colors)
  

### History

def plot_losses_from_array(training_losses, validation_losses):
  epochs = list(range(1, len(training_losses) + 1))

  plt.plot(epochs, training_losses, label='Training Loss',  linestyle='-')

  # Plot validation losses
  plt.plot(epochs, validation_losses, label='Validation Loss', linestyle='-')

  # Add labels and a legend
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.title('Training and Validation Loss Over Epochs')
  plt.legend()

def plot_gan_losses(d_losses,g_losses):
  fig, ax1 = plt.subplots(figsize=(10, 8))

  epoch_count=len(d_losses)

  line1,=ax1.plot(range(1,epoch_count+1),d_losses,label='discriminator_loss',color='orange')
  ax1.set_ylim([0, max(d_losses)])
  ax1.tick_params(axis='y', labelcolor=line1.get_color())
  _=ax1.legend(loc='lower left')

  ax2 = ax1.twinx()
  line2,=ax2.plot(range(1,epoch_count+1),g_losses,label='generator_loss')
  ax2.set_xlim([1,epoch_count])
  ax2.set_ylim([0, max(g_losses)])
  ax2.set_xlabel('Epochs')
  ax2.tick_params(axis='y', labelcolor=line2.get_color())
  _=ax2.legend(loc='upper right')