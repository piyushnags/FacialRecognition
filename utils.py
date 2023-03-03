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


def save_aligned_faces(x: Tensor):
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
    
    path = 'results/aligned_faces'
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