# Built-in Imports
import os, time, io
import argparse
import zipfile
from typing import Any, Tuple, List
from natsort import natsorted
from glob import glob

# Math and Visualization Imports
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

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
            T.Resize( (256,256), antialias=None )
        ])
    

    def __getitem__(self, idx) -> List[Tensor]:
        img = self.dset[idx]
        img = self.preprocess(img)
        if self.transforms is not None:
            img = self.transforms(img)
        return img
    

    def __len__(self) -> int:
        return len(self.dset)



class InceptionDataset(Dataset):
    def __init__(self, root1: str, root2: str):
        super(InceptionDataset, self).__init__()

        if not os.path.exists(root1):
            raise ValueError(f"Path {root1} does not exist")
        
        if not os.path.exists(root2):
            raise ValueError(f"Path {root2} does not exist")
        
        files1 = natsorted(glob(os.path.join(root1, '*.jpg'))
                  + glob(os.path.join(root1, '*.JPG'))
                  + glob(os.path.join(root1, '*.png'))
                  + glob(os.path.join(root1, '*.PNG')))
        
        files2 = natsorted(glob(os.path.join(root1, '*.jpg'))
                  + glob(os.path.join(root1, '*.JPG'))
                  + glob(os.path.join(root1, '*.png'))
                  + glob(os.path.join(root1, '*.PNG')))
        
        self.files1 = files1
        self.files2 = files2
        self.to_tensor = T.ToTensor()
    

    def __getitem__(self, idx) -> Tuple[List[Tensor], List[Tensor]]:
        im1 = Image.open( self.files1[idx] ).convert('RGB')
        im2 = Image.open( self.files2[idx] ).convert('RGB')
        return self.to_tensor(im1), self.to_tensor(im2)
    

    def __len__(self) -> int:
        return len(self.files1)



def get_pre_loader(args: Any) -> DataLoader:
    root = args.data_root
    if not os.path.exists(root):
        raise ValueError(f"Path {root} does not exist")
    
    zip_dataset = ZipDataset(root, cache_into_memory=True)
    augment = None
    if args.add_noise:
        augment = T.Compose([
            _AddNoise(args.noise_var, args.noise_mean)
        ])
    dataset = SUNetDataset(zip_dataset, augment)

    pre_loader = DataLoader(dataset, args.batch_size, num_workers=args.num_workers)
    return pre_loader


def get_loaders(args: Any) -> Tuple[DataLoader]:
    dataset = InceptionDataset(args.root1, args.root2)
    test_batches = (args.num_batches // 11) * args.batch_size
    train_batches = (args.num_batches - (args.num_batches // 11) ) * args.batch_size
    
    train_data, test_data, _ = torch.utils.data.random_split(
        dataset, [train_batches, test_batches, len(dataset) - train_batches - test_batches]
    )
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, num_workers=args.num_workers)

    return train_loader, test_loader