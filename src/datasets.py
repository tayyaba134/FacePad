import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class GeneralSet(Dataset):
    def __init__(self, name=None, sub=None, mode=None, sps=None):
        self.imgs = []
        self.labels = []
        
        # Load and filter the dataset entries based on file existence
        fh = open(f'datasets/{name}/{name}_{sub}.txt', 'r')
        for line in fh:
            line = line.rstrip()
            words = line.split()
            img_path = f'datasets/{name}/rgb/{words[0]}.jpg'
            if os.path.exists(img_path):  # Only add if the file exists
                self.imgs.append(words[0])
                self.labels.append(int(words[1]))
        fh.close()
        
        self.imgs = np.array(self.imgs)
        self.labels = np.array(self.labels)
        
        if mode == 'grand':
            idx = np.where(self.labels != -1)
        elif mode == 'adaptation':
            idx = np.where(self.labels == 0)

        self.root = f'datasets/{name}'
        self.imgs = self.imgs[idx]
        self.labels = self.labels[idx]
        self.transform = transforms.Compose([
            transforms.Resize(sps),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        label = self.labels[index]
        img_name = self.imgs[index]
        img_path = f'{self.root}/rgb/{img_name}.jpg'
        img = Image.open(img_path)
        img = self.transform(img)
        img = (img - 0.5) / 0.5
        return img, label

    def __len__(self):
        return len(self.imgs)

class ClientSet(Dataset):
    def __init__(self, sub, client=None, sps=None):
        self.imgs = []
        self.labels = []
        
        # Load and filter the dataset entries based on file existence
        fh = open(f'datasets/client/{sub}_list.txt', 'r')
        for line in fh:
            line = line.rstrip()
            words = line.split()
            img_path = f'datasets/client/rgb/{sub}/{words[0]}.jpg'
            if os.path.exists(img_path):  # Only add if the file exists
                self.imgs.append(words[0])
                self.labels.append(int(words[2]))
        fh.close()
        
        self.imgs = np.array(self.imgs)
        self.labels = np.array(self.labels)
        self.root = 'datasets/client'
        self.sub = sub
        self.client = client
        self.transform = transforms.Compose([
            transforms.Resize(sps),
            transforms.ToTensor()
        ])

        if client is not None:
            idx = np.where(np.array(words[1], dtype=int) == client)
            self.imgs = self.imgs[idx]
            self.labels = self.labels[idx]

    def __getitem__(self, index):
        label = self.labels[index]
        img_name = self.imgs[index]
        img_path = f'{self.root}/rgb/{self.sub}/{img_name}.jpg'
        img = Image.open(img_path)
        img = self.transform(img)
        img = (img - 0.5) / 0.5
        return img, label

    def __len__(self):
        return len(self.imgs)
