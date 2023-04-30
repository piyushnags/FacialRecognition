# Built-in Imports
import os, time, io
import argparse
import zipfile
from typing import Any, Tuple, List
from natsort import natsorted
from glob import glob
from tqdm import tqdm

# Math and Visualization Imports
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from pytorch_msssim import ssim

# PyTorch Imports
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as T



def evaluate_similarity(model: nn.Module, dataset: Dataset, batch_size: int, num_workers: int, device: Any):
    model.eval()
    loss_fn = nn.MSELoss()

    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    running_ssim = []
    running_mse = []
    for img1, img2 in tqdm(loader):
        img1, img2 = img1.to(device), img2.to(device)

        with torch.no_grad():
            out1 = model(img1)
            out2 = model(img2)
            loss = loss_fn(out1, out2)
            running_mse.append(loss.item())

        score = ssim(img1, img2, data_range=1, size_average=False)
        score = torch.mean(score)
        running_ssim.append(score.item())

    avg_ssim = sum(running_ssim)/len(running_ssim)
    avg_mse = sum(running_mse)/len(running_mse)
    print(f"Average SSIM for Datasets is {avg_ssim:.5f} ") 
    print(f"Average MSE for Datasets is {avg_mse:.5f}")