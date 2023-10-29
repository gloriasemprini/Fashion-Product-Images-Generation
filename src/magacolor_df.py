# %%
from PIL import Image
# import os
# os.environ["OMP_NUM_THREADS"] = '8'
import warnings
warnings.filterwarnings('ignore')
from sklearn.cluster import KMeans
import  matplotlib.pyplot as plt 
import numpy as np

import utils.paths as paths
import utils.df_preprocessing as preprocess
import importlib
importlib.reload(preprocess)
importlib.reload(paths)


def add_megacolor(data_dir, df):

    megacolors = []
    for idx, id in enumerate(df['id']):
        path = data_dir + id
        image = Image.open(path)
        image = image.convert("RGB")

        white_limit = 240
        def is_white(color):
            r, g, b = color
            return r > white_limit and g > white_limit and b > white_limit

        # Initialize a list to store non-white colors
        colors = []

        # Iterate through the image pixels
        for pixel in image.getdata():
            if not is_white(pixel):
                colors.append(pixel)

        kmeans = KMeans(1,  n_init=5=2)  
        kmeans.fit(colors)
        rgb_colors = kmeans.cluster_centers_
        normalized_colors = [(round(r / 255, 2), round(g / 255, 2), round(b / 255, 2)) for r, g, b in rgb_colors]
        megacolor = np.array(normalized_colors).ravel()
        megacolors.append(megacolor)
        # print(str(idx) + " " + id + " " + str(megacolor) + " | megacolors: "  + str(len(megacolors)))
    print("Megacolors size:" + str(len(megacolors)))
    return megacolors

# %%
classes = ["Nail Polish", "Watches", "Sunglasses", "Sarees", "Flip Flops", "Deodorant", "Backpacks"]
df = preprocess.filter_articles(preprocess.get_clean_DF(), classes=classes)
def append_ext(id): return id+".jpg"
df['id'] = df['id'].apply(append_ext)
colors = add_megacolor(paths.IMG_FOLDER, df)
print(colors)

df["megacolors"] = colors
df.to_csv("megacolors5.csv", sep=",")
# %%
