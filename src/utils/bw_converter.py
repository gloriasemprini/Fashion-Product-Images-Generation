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


def saveWithColors(ids, subfolder_name, main_folder=paths.COLOR_IMG_FOLDER):
    if not os.path.exists(main_folder):
        os.makedirs(main_folder)

    if not os.path.exists(main_folder + subfolder_name):
        os.makedirs(main_folder + subfolder_name)

    for id in ids:
        image_file = str(id) + ".jpg"
        input_path = os.path.join(paths.IMG_FOLDER, image_file)
        output_path = os.path.join(main_folder + subfolder_name, image_file)


        Image.open(input_path).save(output_path)
      