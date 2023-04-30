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


def get_pre_loader(root: str) -> DataLoader:
    if not os.path.exists(root):
        raise ValueError(f"Path {root} does not exist")
    
    pass
