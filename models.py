# Built-in Imports
import os, time

# Math and Visualization 
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Torch Imports
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.data import(
    DataLoader,
    Dataset as dset
)

# Torchvision Imports
import torchvision.transforms as T
from torchvision import datasets 

# Model Imports
from torchvision.models import resnet101, ResNet101_Weights

'''
TODO: 
    1. Add support for training
    2. Improve model architecture
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
    def __init__(self, pretrained: bool = False):
        super(Siamese, self).__init__()

        # Load weights if pretrained is true
        weights = None
        if pretrained:
            weights = ResNet101_Weights.IMAGENET1K_V2

        # Get resnet backbone and remove last
        # layer. Add layer for consolidating
        # Siamese inputs
        backbone = resnet101(weights=weights)
        self.out_dim = backbone.fc.in_features
        backbone = nn.Sequential(
            *(list(backbone.children())[:-1])
        )
        self.backbone = backbone
        self.consolidate = nn.Sequential(
            nn.Linear(self.out_dim*2, 512),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, 1)
        )
        self.sigmoid = nn.Sigmoid()

        # Weight initialization for untrained layers
        self.consolidate.apply(self._he_init)
        self.sigmoid.apply(self._he_init)
        if not pretrained: self.backbone.apply(self._he_init) 
    

    def _he_init(self, m: nn.Module):
        '''
        Applies Uniform He Initialization to weights
        Meant to be applied recursively through
        nn.Module.apply()
        '''
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)
            m.bias.data.fill_(0)

    
    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        '''
        Args: 
            x1: input image 1 tensor
            x2: input image 2 tensor

        Output:
            out: Similarity score (Singleton Tensor)
        
        Description:
        Forward propagation of Siamese Network
        '''
        # Pass both inputs through shared backbone
        out1 = self.backbone(x1)
        out2 = self.backbone(x2)

        # Outputs are 4-D tensors so squeeze them to
        # 2-D tensors
        out1, out2 = out1.view(out1.size()[0], -1), out2.view(out2.size()[0], -1)
        out = torch.cat( (out1, out2), dim=1 )

        # Consolidate the branches
        out = self.consolidate(out)
        out = self.sigmoid(out)
        return out


'''
Sandbox for model testing
Currently performs a sanity check on the Siamese 
network by sending in the same image for both inputs.

Expected output: Probability of match
'''
if __name__ == "__main__":
    model = Siamese()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.load_state_dict( torch.load('weights/Siamese_BCE.pth', map_location=device) )
    model.eval()
    
    dataset = datasets.ImageFolder("facenet_pytorch/data/test_images_aligned")
    img = dataset[0][0]
    img_ = dataset[1][0]
    preprocess = T.Compose([
        T.ToTensor()
    ])
    img = preprocess(img).unsqueeze(0)
    img_ = preprocess(img_).unsqueeze(0)
    
    with torch.no_grad():
        p = model(img, img)
    
    print(p)    
    