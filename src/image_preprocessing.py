# %%
CLASSES = ["Watches", "Handbags", "Sunglasses", "Belts", "Backpacks", "Sarees", "Deodorant", "Nail Polish", "Ties", "Flip Flops", "Formal Shoes"]

import importlib
import utils.ploters as ploters 
import random
from PIL import Image
import utils.bw_converter as bw

import analysis 
import utils.paths as paths
import utils.df_preprocessing as df_preprocessing

def get_dataframe_by(df, column_name, value):
    return df[df[column_name] == value]


def get_dataframe_by_article_type(df, article_type):
    return get_dataframe_by(df, "articleType", article_type)

def get_dataframe_by_color(df, color):
    return get_dataframe_by(df, "baseColour", color)




# %% RELOAD
importlib.reload(analysis)
importlib.reload(paths)
importlib.reload(ploters)
importlib.reload(bw)    
importlib.reload(df_preprocessing)

#  %% AD HOC
df = df_preprocessing.get_clean_DF()


# %%
df = df_preprocessing.get_clean_DF()
# df["baseColour"].value_counts()
df_preprocessing.filter_articles(df, ["Sunglasses"])["baseColour"].value_counts()
# %%
my_df = get_dataframe_by(df, "baseColour", "Brown")
ploters.plot_random_image(my_df, num=20)


# %%
analysis.showDatasetDetails(df)

# %% 
analysis.unique_values_for_each_column(df)


# %% Plot random images from dataframe
importlib.reload(ploters)
ids = df['id']
selected_image_ids = random.sample(ids.tolist(),10)
ploters.plot_images_by_id(selected_image_ids)


# %% Loading images into color folder
importlib.reload(ploters)
importlib.reload(bw)

for cl in CLASSES:
    cl_df = get_dataframe_by_article_type(df, cl)
    bw.saveWithColors(cl_df['id'], "subset/subset/")
    ploters.plot_random_image(cl_df, path=paths.COLOR_IMG_FOLDER + "subset/subset/")
# %% Loading images into color folder
importlib.reload(ploters)
importlib.reload(bw)

df = df_preprocessing.filter_articles(df, CLASSES)
for cl in df["baseColour"].unique():
    cl_df = get_dataframe_by_color(df, cl)
    path = paths.DATASET_PATH + "colorati/"
    bw.saveWithColors(cl_df['id'], cl + "/", path)
    print(cl)
    ploters.plot_random_image(cl_df, path=path + cl + "/")


# %%
my_df = get_dataframe_by_article_type(df, "Watches")
# my_df = get_dataframe_by_color(df, "Fluorescent Green")
ploters.plot_random_image(my_df, num=20)

# %%
my_df = get_dataframe_by(df, "baseColour", "Pink")
ploters.plot_random_image(my_df, num=20)

# %%
################################################### Make image square
from PIL import Image, ImageOps
import os
import numpy as np

# Set the paths to your original and destination folders
original_folder = paths.DATASET_PATH + "color/subset/subset/"
destination_folder = paths.DATASET_PATH + "square/"

# Create the destination folder if it doesn't exist
if not os.path.exists(destination_folder):
    os.mkdir(destination_folder)

# Function to add padding to an image
def preprocess_image(image_path):
    # Carica l'immagine in RGB
    image = Image.open(image_path).convert("RGB")
    
    # Ridimensiona in modo proporzionale a 80x80
    image = ImageOps.fit(image, (80, 80), Image.ANTIALIAS)
    
    # Converti in array NumPy e normalizza tra -1 e 1
    image_array = np.array(image).astype(np.float32) / 127.5 - 1
    
    return image_array

# Loop through all the images in the original folder
for filename in os.listdir(original_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(original_folder, filename)
        
        # Applica il preprocessing
        image_array = preprocess_image(image_path)
        
        # Converti di nuovo in immagine PIL per salvarla
        processed_image = Image.fromarray(((image_array + 1) * 127.5).astype(np.uint8))
        
        # Salva l'immagine processata
        destination_path = os.path.join(destination_folder, filename)
        processed_image.save(destination_path)

print("Preprocessing completato e immagini salvate correttamente.")


# %%
