# %%
import pandas as pd
import importlib
import utils.ploters as ploters 
import filters
import random
from PIL import Image
import utils.bw_converter as bw

import analysis 
import utils.paths as paths

# %% RELOAD
importlib.reload(analysis)
importlib.reload(paths)
importlib.reload(ploters)
importlib.reload(filters)

importlib.reload(bw)

# %%
df = pd.read_csv(paths.getDataSetPath('styles.csv'))


# %%
analysis.showDatasetDetails(df)


# %% Plot random images from dataframe
importlib.reload(ploters)
ids = df['id']
selected_image_ids = random.sample(ids.tolist(),10)
ploters.plotImagesById(selected_image_ids)

# %% Plot only random whatches
importlib.reload(filters)
watches_df = filters.get_dataframe_by_article_type(df, "Watches")
#sunglasses_df = filters.get_dataframe_by_article_type(df, "Sunglasses")
# %%
kurtas_df = filters.get_dataframe_by_article_type(df, "Kurtas")
ploters.plotRandomImg(kurtas_df)


# %% WARNING: this code create a new folder with only grayscale watches
importlib.reload(ploters)
importlib.reload(bw)
bw.convert_to_bw(watches_df['id'])
ploters.plotRandomImg(watches_df, path=paths.BW_IMG_FOLDER_INNER)

# %% WARNING 2
importlib.reload(ploters)
importlib.reload(bw)
bw.saveWithColors(watches_df['id'], "watches/")
ploters.plotRandomImg(watches_df, path=paths.COLOR_IMG_FOLDER + "watches/")
#bw.saveWithColors(sunglasses_df['id'], "sunglasses/")
#ploters.plotRandomImg(sunglasses_df, path=paths.COLOR_IMG_FOLDER + "sunglasses/")


# %% Show first image of a directory
import pathlib
data_dir = pathlib.Path(paths.BW_IMG_FOLDER_INNER).with_suffix('')
image_count = len(list(data_dir.glob('*.jpg')))
print(image_count)

l = list(data_dir.glob('*'))
print(l[0])

Image.open(str(l[0]))
