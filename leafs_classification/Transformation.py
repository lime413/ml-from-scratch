import os
import argparse
from PIL import Image, ImageFilter, ImageOps, ImageDraw
import numpy as np
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt
from tqdm import tqdm

def remove_background(image_path):
    img = cv2.imread(image_path)
    img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_green = np.array([25, 40, 20])  # Нижний предел зелёного
    upper_green = np.array([85, 255, 255])  # Верхний предел зелёного
    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  
    largest_component = (labels == largest_label).astype(np.uint8) * 255

    img_rgba[..., 3] = largest_component 

    return img_rgba

def plot_pixel_intensity_distribution(image):
    channels = cv2.split(image)
    color_labels = ['blue', 'green', 'red']

    for i, channel in enumerate(channels):
        hist, bins = np.histogram(channel, bins=256, range=(0, 256), density=True)
        plt.plot(bins[:-1], hist * 100, label=color_labels[i])

    plt.xlabel('Pixel intensity')
    plt.ylabel('Proportion of pixels (%)')
    plt.title('Pixel Intensity Distribution')
    plt.legend()

    plt.show()

def analyze_leaf(image, filename, output_dir, show_graph):
    original_path = os.path.join(output_dir, filename.split('.')[0] + '__original.jpg')
    cv2.imwrite(original_path, image[..., :3])

    if show_graph:
        plot_pixel_intensity_distribution(image[..., :3])

    leaf_mask = image[..., 3]
    gray_img = cv2.cvtColor(image[..., :3], cv2.COLOR_BGR2GRAY)
    damage_mask = cv2.inRange(gray_img, 0, 80)
    damage_mask = cv2.bitwise_and(damage_mask, damage_mask, mask=(leaf_mask > 0).astype(np.uint8))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    damage_mask = cv2.morphologyEx(damage_mask, cv2.MORPH_CLOSE, kernel)
    mask = image.copy()
    mask = mask[..., :3]
    mask[damage_mask < 1] = 255
    cv2.imwrite(os.path.join(output_dir, filename.split('.')[0] + '__mask.JPG'), mask)

    roi = image.copy()
    roi = roi[..., :3]
    roi[np.max(mask, axis=2) != 255] = (0, 255, 0)
    cv2.imwrite(os.path.join(output_dir, filename.split('.')[0] + '__roi.JPG'), roi)

    blurred = cv2.GaussianBlur(damage_mask, (3, 3), 10)  
    cv2.imwrite(os.path.join(output_dir, filename.split('.')[0] + '__blur.jpg'), blurred)

    contours, _ = cv2.findContours(damage_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_img = image[..., :3].copy()
    cv2.drawContours(contours_img, contours, -1, (255, 0, 0), 2)
    cv2.imwrite(os.path.join(output_dir, filename.split('.')[0] + '__contours.jpg'), contours_img)

    heatmap = cv2.applyColorMap(damage_mask, cv2.COLORMAP_JET)
    heatmap = cv2.addWeighted(heatmap, 0.5, image[..., :3], 0.5, 0)
    cv2.imwrite(os.path.join(output_dir, filename.split('.')[0] + '__heatmap.jpg'), heatmap)

def process_directory(datapath, output_dir):
    paths = os.listdir(datapath)
    subpaths = os.listdir(datapath + os.sep + paths[0])
    one_plant = False
    if not os.path.isdir(datapath + os.sep + paths[0] + os.sep + subpaths[0]):
        subpaths = paths
        paths = [(datapath).split('/')[-1]]
        datapath = (datapath).split(paths[0])[0][:-1]
        one_plant = True
    
    for plant in paths:
        subpaths = os.listdir(datapath + os.sep + plant)
        for disease in subpaths:
            if one_plant:
                current_output_dir = os.path.join(output_dir, disease)
            else:
                current_output_dir = os.path.join(output_dir, plant, disease)
            os.makedirs(current_output_dir, exist_ok=True)
            filenames = os.listdir(os.path.join(datapath, plant, disease))
            print('Processing ' + os.path.join(plant, disease))
            for filename in tqdm(filenames):
                temp_image = remove_background(os.path.join(datapath, plant, disease, filename))
                analyze_leaf(temp_image, filename, current_output_dir, show_graph=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, default='leafs_classification/data/images/Apple', help='path to images folder')
    parser.add_argument('--outputpath', type=str, default='leafs_classification/data/images_transformed', help='path to results')

    args = parser.parse_args()

    datapath = args.datapath
    output_dir = args.outputpath
    os.makedirs(output_dir, exist_ok=True)

    if os.path.isfile(datapath):
        temp_image = remove_background(datapath)
        temp_path = os.path.join(output_dir, f'{os.path.basename(datapath).split('.')[0]}.png')
        cv2.imwrite(temp_path, temp_image)
        analyze_leaf(temp_path, output_dir, show_graph=True)
        os.remove(temp_path)
    elif os.path.isdir(datapath):
        process_directory(datapath, output_dir)
    else:
        print('No such directory: ' + datapath)



if __name__ == '__main__':
    main()