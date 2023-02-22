import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as T

'''
TODO: 
    1. Create class for Siamese Network using
    Resnet50 as backbone with pretrained weights.

    2. Use JPEG encodings as input instead of raw
    images.

    3. Use weights from Transfer Learning to study
    accuracy    
'''

'''
Class for Siamese Network built using
ResNet50 as a backbone with pretrained
weights.

Input:
    x1: first image tensor
    x2: second image tensor

Output:
    pred: 1 for match, 0 for no match

Loss Function:
    Triplet loss is used. Anchor and positive
    sample don't need to be mapped to the same
    point in the vector space, useful for generating
    meaningful positive matches in tasks like
    facial recognition.
'''
class Siamese(nn.Module):
    def __init__(self):
        super(Siamese, self).__init__()
    
    def forward(x1, x2):
        raise NotImplementedError