from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import cv2
import torch

class LeafsDataset(Dataset):    
    def __init__(self, datapath, attribute, train, transform):
        super().__init__()

        if train:
            proportion = 0.8
        else:
            proportion = 0.2

        self.diseases = sorted(os.listdir(datapath))
        self.images = {}
        for i, disease in enumerate(self.diseases):
            files = np.array([f for f in os.listdir(os.path.join(datapath, disease)) if f.endswith(attribute + '.jpg')])
            idxs = np.random.random_integers(0, len(files)-1, int(len(files)*proportion))
            files = files[idxs]
            self.images[i] = files

        self.lens = np.array([len(val) for val in self.images.values()], dtype=np.int32)
        self.datapath = datapath
        self.transform = transform


    def __len__(self):
        return np.sum(self.lens)
    
    def __getitem__(self, idx):
        found = False
        disease = 0
        while not found:
            if idx >= np.sum(self.lens[:disease+1]):
                disease += 1
            else:
                found = True
        
        idx = int(idx - np.sum(self.lens[:disease]))

        image = cv2.imread(os.path.join(self.datapath, self.diseases[disease], self.images[disease][idx]), cv2.IMREAD_COLOR_RGB)
        image = (image.astype(np.float32)) / 255.0
        # image = image.transpose((2, 0, 1))

        tensor_image = self.transform(image)
        tensor_label = torch.tensor(disease)

        return (tensor_image, tensor_label)