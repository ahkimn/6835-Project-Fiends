import os
import cv2
import csv

RGB_NAME_FILE = 'color.txt'
RGB_IMG_DIR = 'rgb'
DEPTH_NAME_FILE = 'depth.txt'
DEPTH_IMG_DIR = 'depth'
JOINT_FILE = 'joints.csv'
TIME_FILE = 'timestamps.csv'


def load_rgbd(directory, data=''):

    if data == 'rgb':
        name_file = RGB_NAME_FILE
        img_dir = RGB_IMG_DIR
    elif data == 'depth':
        name_file = DEPTH_NAME_FILE
        img_dir = DEPTH_IMG_DIR
    else:
        raise ValueError

    name_file = os.path.join(directory, name_file)

    assert(os.path.isfile(name_file))

    name_file = open(name_file, 'r')
    img_files = name_file.readlines()
    img_data = []

    for f in img_files:
        img_file = os.path.join(directory, img_dir, f.strip())
        # img_file = directory + "/" + img_dir + "/" + f.strip()

        if data == 'rgb':
            img = cv2.flip(cv2.imread(img_file), 1)
        else:
            img = cv2.flip(cv2.imread(img_file, cv2.IMREAD_ANYDEPTH), 1)

        img_data.append(img)

    name_file.close()
    return img_data


def load_csv(directory, data=''):

    if data == 'joint':
        data_file = JOINT_FILE

    elif data == 'timestamp':
        data_file = TIME_FILE

    else:
        raise ValueError

    data_file = os.path.join(directory, data_file)

    assert(os.path.isfile(data_file))

    data_file = open(data_file, 'r')
    reader = csv.reader(data_file)

    data = [[int(i) for i in row if i != ''] for row in reader]
    data_file.close()

    return data
