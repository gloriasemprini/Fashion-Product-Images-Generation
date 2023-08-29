DATASET_PATH = "../dataset/"

IMG_FOLDER = DATASET_PATH + "images/images/"
BW_IMG_FOLDER = DATASET_PATH + "bw/"
BW_IMG_FOLDER_INNER = BW_IMG_FOLDER + "img/"

COLOR_IMG_FOLDER = DATASET_PATH + "color/"

def getDataSetPath(file_path):
    return DATASET_PATH + file_path

def getImagePath(image_id, folder=IMG_FOLDER):
    return folder + str(image_id) + ".jpg"