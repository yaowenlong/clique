import os,sys
import torch.utils
import torch
from torch.utils.data.dataloader import DataLoader
from PIL import Image
import numpy as np
class MydataSet(torch.utils.data.Dataset):

    def __init__(self,layout, imgpath, transform=None, target_transform=None):
        imgs = []
        for line in open(layout):
            
            line = line.split(" ")
            if line != ['\n']:
                fn = line[0].split("/", 5)
                label = line[1].replace("\n", '')
                label = int(label)
                imgs.append((fn[5], label))
            
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.imgpath = imgpath
        
    def __getitem__(self, index):
        fn, label = self.imgs[index]
        label = torch.tensor([label])
        img = Image.open(self.imgpath+fn)
        if self.transform is not None:
            img = self.transform(img)
            np.transpose(img, (1, 2, 0))
        
        return img, label

    def __len__(self):
        return len(self.imgs)

    @staticmethod
    def collate_fn(batch):
        batch.sort(key=lambda x: len(x[1]), reverse=True)
        img, label = zip(*batch)
        pad_label = []
        lens = []
        max_len = len(label[0])
        for i in range(len(label)):
            temp_label = [0] * max_len
            temp_label[:len(label[i])] = label[i]
            pad_label.append(temp_label)
            lens.append(len(label[i]))
        return img, pad_label, lens