DATASET_PATH = "../dataset/"

IMG_FOLDER = DATASET_PATH + "images/"
BW_IMG_FOLDER = DATASET_PATH + "bw/"
BW_IMG_FOLDER_INNER = BW_IMG_FOLDER + "img/"

COLOR_IMG_FOLDER = DATASET_PATH + "color/"

def get_dataset_folder_file_path(file_name):
    return DATASET_PATH + file_name

def get_image_path(image_id, folder=IMG_FOLDER):
    return folder + str(image_id) + ".jpg"