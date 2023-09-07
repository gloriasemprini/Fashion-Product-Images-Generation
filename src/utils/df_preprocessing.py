import pandas as pd
import utils.paths as paths

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
 
CLASSES = ["Sunglasses"]

def filter_articles(df, classes=CLASSES):
    return df[df['articleType'].isin(classes)]

def get_clean_DF():
    df = pd.read_csv(paths.getDataSetPath('styles.csv'), dtype=str)
    df = df[df.notnull()["baseColour"]] # remove null values from basecolor column
    black_list = [35467, 36105, 24501, 24502, 24504, 5623, 10804, 12529, 17028, 17029,17736, 23826, 24768, 35575, 35632, 39954, 47084, 5851, 5852, 12348,
                  37408,37409, 37410, 39374, 39375, 39376,39377, 44998, 45017, 48014, 48015, 50400, 50401, 50402, 58446, 59282]
    black_list = [str(x) for x in black_list]
    df = df.drop(df[df["id"].isin(black_list)].index)

    ## Colour aggregation
    baseColour = "baseColour"
    df[baseColour] = df[baseColour].replace("Lime Green", "Green")
    df[baseColour] = df[baseColour].replace("Fluorescent Green", "Green")
    df[baseColour] = df[baseColour].replace("Sea Green", "Green")
    df[baseColour] = df[baseColour].replace("Teal", "Green")

    df[baseColour] = df[baseColour].replace("Taupe", "Grey")
    df[baseColour] = df[baseColour].replace("Grey Melange", "Grey")
    df[baseColour] = df[baseColour].replace("Steel", "Grey")
    df[baseColour] = df[baseColour].replace("Silver", "Grey")

    df[baseColour] = df[baseColour].replace("Mushroom Brown", "Brown")
    df[baseColour] = df[baseColour].replace("Nude", "Brown")
    df[baseColour] = df[baseColour].replace("Coffee Brown", "Brown")
    df[baseColour] = df[baseColour].replace("Burgundy", "Brown")
    df[baseColour] = df[baseColour].replace("Copper", "Brown")
    df[baseColour] = df[baseColour].replace("Bronze", "Brown")
    df[baseColour] = df[baseColour].replace("Tan", "Brown")
    df[baseColour] = df[baseColour].replace("Khaki", "Brown")

    df[baseColour] = df[baseColour].replace("Rose", "Red")
    df[baseColour] = df[baseColour].replace("Orange", "Red")
    df[baseColour] = df[baseColour].replace("Rust", "Red")
    df[baseColour] = df[baseColour].replace("Maroon", "Red")

    df[baseColour] = df[baseColour].replace("Magenta", "Pink")
    df[baseColour] = df[baseColour].replace("Peach", "Pink")

    df[baseColour] = df[baseColour].replace("Mauve", "Purple")
    df[baseColour] = df[baseColour].replace("Lavender", "Purple")

    df[baseColour] = df[baseColour].replace("Metallic", "Black")
    df[baseColour] = df[baseColour].replace("Charcoal", "Black")

    df[baseColour] = df[baseColour].replace("Turquoise Blue", "Blue")

    df[baseColour] = df[baseColour].replace("Mustard", "Yellow")
    df[baseColour] = df[baseColour].replace("Gold", "Yellow")

    df[baseColour] = df[baseColour].replace("Off White", "White")

    df[baseColour] = df[baseColour].replace("Skin", "Beige")

    df[baseColour] = df[baseColour].replace("Cream", "Multi")
    df[baseColour] = df[baseColour].replace("Olive", "Multi")
    df[baseColour] = df[baseColour].replace("Navy Blue", "Multi")
    return df