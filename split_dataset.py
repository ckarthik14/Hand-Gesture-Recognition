import os
import random
from shutil import copyfile
from tqdm import tqdm

SEED = 1
TRAIN = 0.8
PATH = 'dataset/'
TRAIN_DEST = 'train/'
TEST_DEST = 'test/'

class_num = {'yes':'1', 'no':'2', 'nothing':'3'}

# Getting 3 letter id in string
def get_id(i):
    return str(int(i/100))+str(int((i/10) % 10))+str(int(i%10))

for image_class in tqdm(os.listdir(PATH)):
    IMAGE_PATH = PATH + image_class
    num_of_images = len(os.listdir(IMAGE_PATH))

    separator_list = [i for i in range(num_of_images)]

    random.seed(SEED)
    random.shuffle(separator_list)

    img_list = []

    for img in os.listdir(IMAGE_PATH):
        img_list.append(img)

    # Getting first 70% of indices
    train_indices = int(TRAIN * num_of_images)

    # Copying to respective directories
    for i in range(train_indices):
        img_id = get_id(i)
        copyfile(PATH + image_class + '/' + img_list[separator_list[i]], TRAIN_DEST + img_id + '_' + class_num[image_class] + '.jpg')

    for i in range(train_indices, num_of_images):
        img_id = get_id(i - train_indices)
        copyfile(PATH + image_class + '/' + img_list[separator_list[i]], TEST_DEST + img_id + '_' + class_num[image_class] + '.jpg')