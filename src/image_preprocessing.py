# %%
import pandas as pd
import importlib
import utils.ploters as ploters 
import random
from PIL import Image
import utils.bw_converter as bw

import analysis 
import utils.paths as paths

# %% RELOAD
importlib.reload(analysis)
importlib.reload(paths)
importlib.reload(ploters)

importlib.reload(bw)

# %%
df = pd.read_csv(paths.getDataSetPath('styles.csv'))

def get_dataframe_by(df, value, column_name):
    list = []
    for i, type in enumerate(df[column_name]):
        if(type == value):
           list.append(df.iloc[i])
    return pd.DataFrame(list)


def get_dataframe_by_article_type(df, article_type):
    return get_dataframe_by(df, article_type, "articleType")

def get_dataframe_by_color(df, color):
    return get_dataframe_by(df, color, "baseColour")
# %%
analysis.showDatasetDetails(df)

# %% 
analysis.unique_values_for_each_column(df)


# %% Plot random images from dataframe
importlib.reload(ploters)
ids = df['id']
selected_image_ids = random.sample(ids.tolist(),10)
ploters.plotImagesById(selected_image_ids)

#### Possible classes:

# "Watches" #2542
# "Handbags" #1759
# "Sunglasses" #1073
# "Wallets" #936
# "Belts" #813
# "Backpacks" #724
# "Socks" #686
# "Perfume and Body Mist" #614

### shoes
# "Casual Shoes" #2846
# "Sports Shoes" #2036
# "Heels" #1323
# "Flip Flops" #916
# "Formal Shoes" #637
# "Flats" #500

### So so
# "Trousers" #530
 
# %% Loading images into color folder
classes = ["Wallets", "Heels", "Perfume and Body Mist", "Sunglasses"]
importlib.reload(ploters)
importlib.reload(bw)

for cl in classes:
    cl_df = get_dataframe_by_article_type(df, cl)
    bw.saveWithColors(cl_df['id'], cl + "/")
    ploters.plotRandomImg(cl_df, path=paths.COLOR_IMG_FOLDER + cl + "/")


# %%
# my_df = get_dataframe_by_article_type(df, "Trousers")
my_df = get_dataframe_by_color(df, "Fluorescent Green")
ploters.plotRandomImg(my_df, num=5)

# %%
my_df = get_dataframe_by(df, "", "baseColour")
print(my_df)
# ploters.plotRandomImg(my_df, num=20)

# %%
