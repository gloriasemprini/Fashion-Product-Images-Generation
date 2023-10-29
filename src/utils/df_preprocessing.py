import pandas as pd
import utils.paths as paths


# CLASSES = ["Watches", "Handbags", "Sunglasses", "Belts", "Flip Flops",
#             "Backpacks", "Sarees", "Deodorant", "Nail Polish", "Ties"] 
# CLASSES = ["Sports Shoes", "Formal Shoes", "Ties", "Nail Polish", "Deodorant", "Sarees"] 
def filter_articles(df, classes):
    return df[df['articleType'].isin(classes)]

def get_clean_DF():
    df = pd.read_csv(paths.get_dataset_folder_file_path('megacolors5.csv'), dtype=str)
    df = df[df.notnull()["baseColour"]] # remove null values from basecolor column
    #black list contains id of articles with wrong article type or with other problems
    black_list = [35467, 36105, 24501, 24502, 24504, 5623, 10804, 12529, 17028, 17029,17736, 23826, 24768, 35575, 35632, 39954, 
                  47084, 5851, 5852, 12348, 37408,37409, 37410, 39374, 39375, 39376,39377, 44998, 45017, 48014, 48015, 50400, 
                  50401, 50402, 58446, 59282, 34774,34777, 51301, 47145,47146,34776,34781,47147,58157, 34773, 34888,34779,34780,
                  34782,47138,47140,47142,47144, 51271, 51272, 1790, 6125, 8217, 9573, 9911, 6896, 1535, 1566, 1577, 1581, 4451,
                  55004, 19222, 49690, 49699, 12345, 23558, 23559, 23593, 26735, 26781, 26785, 32528, 32531, 32549, 32550, 41904, 
                  43098, 43099, 43100, 44949, 45719, 45720, 45721, 52871, 31847, 36802, 51519, 51522, 51523, 51524, 54651, 54652,
                  56342, 32530, 5364, 5365, 4363, 5582, 5593, 24459, 5381, 5380, 5402,
                  7217,]  + [i+1 for i in range(37876, 37896)] + [i+1 for i in range(21571, 21575)] + [i+1 for i in range(46896, 46901)]
 
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

    df[baseColour] = df[baseColour].replace("Skin", "Beige")

    df[baseColour] = df[baseColour].replace("Grey", "White") # White + Grey
    df[baseColour] = df[baseColour].replace("Beige", "White") # White + Beige
    df[baseColour] = df[baseColour].replace("Off White", "White")

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

    df[baseColour] = df[baseColour].replace("Cream", "Multi")
    df[baseColour] = df[baseColour].replace("Olive", "Multi")
    df[baseColour] = df[baseColour].replace("Navy Blue", "Multi")
    return df