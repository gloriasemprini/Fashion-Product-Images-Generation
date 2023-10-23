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
df_preprocessing.filter_articles(df, ["Watches", "Sunglasses"])["baseColour"].value_counts()
# %%
my_df = get_dataframe_by(df, "baseColour", "Yellow")
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
    bw.saveWithColors(cl_df['id'], "subset2/subset/")
    ploters.plot_random_image(cl_df, path=paths.COLOR_IMG_FOLDER + "subset2/subset/")
# %% Loading images into colorrrrrrrrrrr folder
importlib.reload(ploters)
importlib.reload(bw)

df = df_preprocessing.filter_articles(df)
for cl in df["baseColour"].unique():
    cl_df = get_dataframe_by_color(df, cl)
    path = paths.DATASET_PATH + "colorati/"
    bw.saveWithColors(cl_df['id'], cl + "/", path)
    print(cl)
    ploters.plot_random_image(cl_df, path=path + cl + "/")


# %%
my_df = get_dataframe_by_article_type(df, "Trousers")
# my_df = get_dataframe_by_color(df, "Fluorescent Green")
ploters.plot_random_image(my_df, num=20)

# %%
my_df = get_dataframe_by(df, "baseColour", "Rose")
ploters.plot_random_image(my_df, num=20)

# %%
################################################### Make image square
from PIL import Image
import os

# Set the paths to your original and destination folders
original_folder = paths.DATASET_PATH + "subset/subset/"
destination_folder = paths.DATASET_PATH + "square/"

# Create the destination folder if it doesn't exist
if not os.path.exists(destination_folder):
    os.mkdir(destination_folder)

# Function to add padding to an image
def add_padding_to_image(image_path, padding_size):
    original_image = Image.open(image_path)
    
    # Create a new image with the desired size and white background
    padded_image = Image.new("RGB", (80, 80), (255, 255, 255))
    
    # Paste the original image in the center of the new image
    padded_image.paste(original_image, (10, 0))
    
    return padded_image

# Loop through all the images in the original folder
for filename in os.listdir(original_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(original_folder, filename)
        
        # Add padding to the image
        padded_image = add_padding_to_image(image_path, 10)
        
        # Save the padded image to the destination folder
        destination_path = os.path.join(destination_folder, filename)
        padded_image.save(destination_path)

print("Padding and saving complete.")



# %%
