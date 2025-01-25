import os
import argparse
import matplotlib.pyplot as plt
import numpy as np


def plot(args):
    datapath = args.datapath
    paths = os.listdir(datapath)
    subpaths = os.listdir(datapath + os.sep + paths[0])
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e']
    if not os.path.isdir(datapath + os.sep + paths[0] + os.sep + subpaths[0]):
        subpaths = paths
        paths = [(args.datapath).split('/')[-1]]
        datapath = (args.datapath).split(paths[0])[0][:-1]
    
    for plant in paths:
            labels = []
            counts = []
            subpaths = os.listdir(datapath + os.sep + plant)
            for disease in subpaths:
                labels.append(disease)
                counts.append(len(os.listdir(datapath + os.sep + plant + os.sep + disease)))
            fig = plt.figure(figsize=(15,5))
            plt.subplot(121)
            plt.pie(counts, labels=labels, colors=colors[:len(counts)])
            plt.axis('equal')
            plt.subplot(122)
            plt.bar(labels, counts, color=colors[:len(counts)])

            fig.suptitle(plant + ' class distribution')

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-datapath', type=str, default='leafs_classification/images',
                        help='datapath for leafs images')
    plot(parser.parse_args())