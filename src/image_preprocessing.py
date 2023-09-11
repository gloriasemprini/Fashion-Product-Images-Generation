# %%

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

for cl in ["Watches", "Sunglasses"]:
    cl_df = get_dataframe_by_article_type(df, cl)
    bw.saveWithColors(cl_df['id'], cl + "/")
    ploters.plot_random_image(cl_df, path=paths.COLOR_IMG_FOLDER + cl + "/")
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


