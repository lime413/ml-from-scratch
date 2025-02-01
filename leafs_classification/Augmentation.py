import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm
import random

def flip_image(image, file_path):
    flip = cv2.flip(image, 1)
    cv2.imwrite(file_path.split('.')[0] + '_flip.JPG', flip)

def rotate_image(image, file_path):
    rotate = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(file_path.split('.')[0] + '_rotate.JPG', rotate)

def skew_image(image, file_path):
    rows, cols = image.shape[:2]
    pts1 = np.float32([[cols*.25, rows*.95], [cols*.90, rows*.95], [cols*.10, 0], [cols, 0]])
    pts2 = np.float32([[cols*0.1, rows], [cols, rows], [0, 0], [cols, 0]])    
    M = cv2.getPerspectiveTransform(pts1,pts2)
    skew = cv2.warpPerspective(image, M, (cols, rows))
    cv2.imwrite(file_path.split('.')[0] + '_skew.JPG', skew)

def scale_image(image, file_path):
    rows, cols = image.shape[:2]
    n = np.random.randint(10, 50)
    scale = image[n:-n, n:-n]
    scale = cv2.resize(scale, (rows, cols))
    cv2.imwrite(file_path.split('.')[0] + '_shear.JPG', scale)

def contrast_image(image, file_path):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl,a,b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    contrast = np.hstack((image, enhanced_img))
    cv2.imwrite(file_path.split('.')[0] + '_contrast.JPG', contrast)

def illuminate_image(image, file_path, gamma=1.5):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    illuminate = cv2.LUT(image, table)
    cv2.imwrite(file_path.split('.')[0] + '_illuminate.JPG', illuminate)

def augment_image(file_path, output_dir, n_adds):    
    image = cv2.imread(file_path)
    methods = np.array([illuminate_image, flip_image, skew_image, scale_image, rotate_image, contrast_image])
    idxs = random.sample(range(0, len(methods)-1), n_adds)
    chosed_methods = methods[idxs]

    for method in chosed_methods:
        method(image, os.path.join(output_dir, file_path.split('/')[-1]))

def augment_dataset(datapath):
    paths = os.listdir(datapath)
    subpaths = os.listdir(datapath + os.sep + paths[0])
    if not os.path.isdir(datapath + os.sep + paths[0] + os.sep + subpaths[0]):
        subpaths = paths
        paths = [(datapath).split('/')[-1]]
        datapath = (datapath).split(paths[0])[0][:-1]

    for plant in paths:
        subpaths = os.listdir(os.path.join(datapath, plant))
        target_count = int(np.max([len(os.listdir(os.path.join(datapath, plant, disease))) for disease in subpaths]))

        for disease in subpaths:
            current_dir = os.path.join(datapath, plant, disease)
            filenames = os.listdir(current_dir)
            current_count = len(filenames)
            print(f'Processing "{current_dir}". Found: {current_count}. Target: {target_count}')

            if current_count >= target_count:
                print('Done')
                continue
            
            n_adds = int(target_count / current_count) - 1
            n_adds_1 = target_count - current_count * (n_adds + 1)

            filenames = sorted(filenames, key=lambda f: int(f.split(')')[0].split('(')[1]))
            for i in range(0, n_adds_1):
                augment_image(os.path.join(current_dir, filenames[i]), current_dir, n_adds + 1)
            for i in range(n_adds_1, len(filenames)):
                augment_image(os.path.join(current_dir, filenames[i]), current_dir, n_adds)

            print('Done')

if __name__ == '__main__':
    np.random.seed(2025)

    parser = argparse.ArgumentParser()
    parser.add_argument('-datapath', type=str, default='leafs_classification/data/images', help='path to data')
    args = parser.parse_args()

    augment_dataset(args.datapath)