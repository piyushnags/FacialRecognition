# Buit-in Imports
import os, time, typing

# Math Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Imaging and Video Imports
from PIL import Image
import cv2

# PyTorch Imports
import torch
from torch import Tensor
import torchvision

# Facenet Imports
from facenet_pytorch import *



class AddNoise():
    '''
    Description:
        Callable for adding Gaussian noise
        By default, Gaussian noise with zero mean
        and unit variance is added

    Args:
        var: variance of Gaussian noise
        mean: mean of Gaussian noise
    
    Returns:
        x: corrupted image(s)
    '''
    def __init__(self, var=1., mean=0.):
        self.std = var**0.5
        self.mean = mean


    def __call__(self, x: Tensor) -> Tensor:
        x += self.std*torch.randn(x.size()) + self.mean
        return torch.clamp(x, 0, 1)



def save_faces(x: Tensor, path: str):
    '''
    Description:
    Utility function to save aligned faces extracted
    from video frames using the MTCNN model.

    Args:
        x (Tensor): Tensor batch containing all detected
                    faces from a particular frame
    
    Returns:
        None
        Saves enumerated images in a fixed 
        directory (results/aligned_faces)

    See Also:
        mtcnn.py: fixed_image_standardization()
    '''
    if x is None:
        print("Received NoneType as Input, cannot save NoneType")
        return
    
    path = os.path.join('results', path)
    if not os.path.exists(path):
        os.makedirs(path)

    # Iterate through all faces in the batch
    # Convert the faces to uint8 representation
    for i in range(x.size()[0]):
        tmp = torch.round(x[i]*128 + 127.5)
        tmp = tmp.permute(1,2,0)
        tmp = tmp.numpy().astype(np.uint8)
        plt.imsave(
            os.path.join(path, 'face_{}.png'.format(i+1)),
            tmp
        )