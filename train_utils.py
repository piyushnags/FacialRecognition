# Built-in Imports
import os, time, io
import argparse
import zipfile
from typing import Any, Tuple, List

# Math and Visualization Imports
import numpy as np
import matplotlib.pyplot as plt
import cv2

# PyTorch Imports
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as T



class _AddNoise():
    def __init__(self, var=1., mean=0.):
        self.std = var**0.5
        self.mean = mean


    def __call__(self, x: Tensor) -> Tensor:
        x += self.std*torch.randn(x.size()) + self.mean
        return torch.clamp(x, 0, 1)



class ZipDataset(Dataset):
    def __init__(self, root_path, cache_into_memory=False):
        if cache_into_memory:
            f = open(root_path, 'rb')
            self.zip_content = f.read()
            f.close()
            self.zip_file = zipfile.ZipFile(io.BytesIO(self.zip_content), 'r')
        else:
            self.zip_file = zipfile.ZipFile(root_path, 'r')
            
        self.name_list = list(filter(lambda x: x[-4:] == '.jpg', self.zip_file.namelist()))
        self.to_tensor = T.ToTensor()

    def __getitem__(self, key):
        buf = self.zip_file.read(name=self.name_list[key])
        img = cv2.imdecode(np.frombuffer(buf, dtype=np.uint8), cv2.IMREAD_COLOR)
        img = self.to_tensor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return img

    def __len__(self):
        return len(self.name_list)
    


class SUNetDataset(Dataset):
    def __init__(self, zip_dataset, transforms=None):
        super(SUNetDataset, self).__init__()
        self.dset = zip_dataset
        self.transforms = transforms
        self.preprocess = T.Compose([
            T.Resize( (320,320), antialias=None )
        ])
    

    def __getitem__(self, idx) -> List[Tensor]:
        img = self.dset[idx]
        img = self.preprocess(img)
        if self.transforms is not None:
            img = self.transforms(img)
        return img
    

    def __len__(self) -> int:
        return len(self.dset)



def get_pre_loader(args: Any) -> DataLoader:
    root = args.data_root
    if not os.path.exists(root):
        raise ValueError(f"Path {root} does not exist")
    
    zip_dataset = ZipDataset(root, cache_into_memory=True)
    augment = T.Compose([
        _AddNoise(args.noise_var, args.noise_mean)
    ])
    dataset = SUNetDataset(zip_dataset, augment)

    pre_loader = DataLoader(dataset, args.batch_size, num_workers=args.num_workers)
    return pre_loader



