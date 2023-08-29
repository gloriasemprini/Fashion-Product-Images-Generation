import utils.paths as paths
import os
from PIL import Image


def convert_to_bw(ids):  
    if not os.path.exists(paths.BW_IMG_FOLDER):
        os.makedirs(paths.BW_IMG_FOLDER)

    if not os.path.exists(paths.BW_IMG_FOLDER + "img/"):
        os.makedirs(paths.BW_IMG_FOLDER + "img/")
    
    # bw_files = os.listdir(paths.BW_IMAGES)
    # print(bw_files)
    for id in ids:
        image_file = str(id) + ".jpg"
        input_path = os.path.join(paths.IMG_FOLDER, image_file)
        output_path = os.path.join(paths.BW_IMG_FOLDER + "img/", image_file)


        image = Image.open(input_path)
        bw_image = image.convert('L')
        bw_image.save(output_path)


def saveWithColors(ids, subfolder_name):
    if not os.path.exists(paths.COLOR_IMG_FOLDER):
        os.makedirs(paths.COLOR_IMG_FOLDER)

    if not os.path.exists(paths.COLOR_IMG_FOLDER + subfolder_name):
        os.makedirs(paths.COLOR_IMG_FOLDER + subfolder_name)

    for id in ids:
        image_file = str(id) + ".jpg"
        input_path = os.path.join(paths.IMG_FOLDER, image_file)
        output_path = os.path.join(paths.COLOR_IMG_FOLDER + subfolder_name, image_file)


        Image.open(input_path).save(output_path)
      