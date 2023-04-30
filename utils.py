# Buit-in, Parsing Imports
from typing import Any, List
import os, time
import argparse
import yaml
from glob import glob
from collections import OrderedDict

# File handling Imports
from natsort import natsorted

# Math Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Imaging and Video Imports
from PIL import Image
import cv2
from skimage import img_as_ubyte

# PyTorch Imports
import torch
import torch.nn as nn
from torch import Tensor
import torchvision
import torchvision.transforms.functional as TF

# Facenet Imports
from facenet_pytorch import *

# SUNet imports 
from models import SUNet_model



class AddNoise():
    '''
    Description:
        Callable for adding Gaussian noise to a standardized img
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
        x_min = (0 - 127.5)/128.0
        x_max = (255 - 127.5)/128.0
        return torch.clamp(x, x_min, x_max)



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


def get_options(path: str) -> object:
    '''
    Description:
        Wrapper function to get options from
        a YAML config file
    
    Args:
        path: string with path to YAML config
              file
    
    Returns:
        options: object containing parsed options
                 from YAML config file
    '''
    with open(path, 'r') as config:
        options = yaml.safe_load(config)
    
    return options


def save_img(filepath, img):
    '''
    Description:
        Wrapper function to save an image using openCV
    
    Args:
        filepath: string contaning save location
        img: numpy array of image
    
    Returns:
        None
    '''
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def parse_options() -> Any:
    '''
    Description:
        helper function for parsing user input args
    
    Args:
        User input
    
    Returns:
        args: namespace object
    '''
    parser = argparse.ArgumentParser()

    # SUNet options
    parser.add_argument('--yaml_path', type=str, default='training.yaml', help='Path to YAML config, needed for SUNet configuration')
    parser.add_argument('--input_dir', type=str, default='results/noisy_faces/', help='path to dir containing corrupted images')
    parser.add_argument('--result_dir', type=str, default='sunet_results/', help='path to save dir for storing restored images')
    parser.add_argument('--weights', type=str, default='weights/model.pth', help='path to SUNet transformer weights')
    parser.add_argument('--window_size', type=int, default=8, help='Shifting window size')
    parser.add_argument('--device', type=str, default='cpu', help='Device for training/inference. Use cuda for GPU')
    parser.add_argument('--noise_var', type=float, default=3e-2, help='Variance of Gaussian noise added as corruption')
    parser.add_argument('--noise_mean', type=float, default=1e-2, help='Mean of Gaussian noise added as corruption')

    # General Options
    parser.add_argument('--num_workers', type=int, default=2, help='Number of worker threads during loading')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size during loading')
    parser.add_argument('--data_root', type=str, default='data_root/', help='Path to zipped dataset')

    # Training Options
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate for training')
    parser.add_argument('--optim', type=str, default='adam', help='optimizer used during training')
    parser.add_argument('--scheduler', type=str, default='step', help='Scheduler for adaptive learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay to prevent weight explosion')
    parser.add_argument('--step_size', type=int, default=3, help='step size for step lr scheduler')
    parser.add_argument('--gamma', type=float, default=0.975, help='Decay for step LR scheduler')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of Epochs to train')
    parser.add_argument('--num_batches', type=int, default=330, help='Total training batches for training and validation split as 10:1')
    parser.add_argument('--train_dir', type=str, default='train_results/', help='Dir to store training results')
    parser.add_argument('--print_freq', type=int, default=10, help='Print frequency during training')

    # Blurry dataset Options
    parser.add_argument('--num_blurry_batches', type=int, default=44, help='Number of batches to be generated for training InceptionResnet')
    parser.add_argument('--root1', type=str, default='data1/', help='Path to blurry image dataset')
    parser.add_argument('--root2', type=str, default='data2/', help='Path to clean images')

    # Flags
    parser.add_argument('--train', action='store_true', help='Flag to run training script')
    parser.add_argument('--make_blurry', action='store_true', help='Flag to generate the blurry dataset')
    parser.add_argument('--save_aligned_ds', action='store_true', help='Flag to create a dataset with only faces. Needs pre-existing dataset')
    parser.add_argument('--add_noise', action='store_true', help='Flag to enable noise when sampling data')
    args = parser.parse_args()
    return args


def load_checkpoint(model: nn.Module, weights: str, device: Any):
    '''
    Description:
        helper function for loading transformer weights.
        Improved version of the function in the original repo
    
    Args:
        model: torch model, transformer
        weights: string with path to weights
        device: cpu/cuda for loading weights
    
    Returns:
        None
    '''
    # add device in map location to ensure that model and data
    # are on correct devices
    checkpoint = torch.load(weights, map_location=device)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    
    # Bit of a hack right here, but doesn't happen in 
    # most cases
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)


def clean_images(inp_dir: str, out_dir: str, weights: str, opt: object, device: Any):
    '''
    Description:
        Helper function that calls SUNet transformer to clean all
        corrupted images within a directory
    
    Args:
        inp_dir: string with path to input directory
        out_dir: string containin save path
        weights: string with path to model weights
        opt: object containing SUNet options
        device: cpu/cuda for inference
    
    Returns:
        None
    '''
    # Create output dir if it does not exist
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    files = natsorted(glob(os.path.join(inp_dir, '*.jpg'))
                  + glob(os.path.join(inp_dir, '*.JPG'))
                  + glob(os.path.join(inp_dir, '*.png'))
                  + glob(os.path.join(inp_dir, '*.PNG')))
    
    if len(files) == 0:
        raise RuntimeError(f"No files at {inp_dir}")
    
    # Instantiate transformer model and move to device
    model = SUNet_model(opt)
    model.to(device)

    # Get weights and prepare for inference
    load_checkpoint(model, weights, device)
    model.eval()

    print('Calling SUNet Transformer...')

    # Clean 'for each' file
    for f in files:
        # Need to convert to RGB
        img = Image.open(f).convert('RGB')

        img_ = TF.to_tensor(img).unsqueeze(0).to(device)

        with torch.no_grad():
            restored = model(img_)    
            restored = torch.clamp(restored, 0, 1)
            restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
        
        restored = img_as_ubyte(restored[0])
        f_ = os.path.splitext(os.path.split(f)[-1])[0]
        save_img((os.path.join(out_dir, f_ + '.png')), restored)
    
    print('Success!')


def evaluate_embeddings(e1: Tensor, e2: Tensor) -> pd.DataFrame:
    '''
    Description:
        helper function to evaluate embeddings based on 
        L2 norm

        Args:
            e1: first embedding (Tensor)
            e2: second embedding (Tensor)
        
        Returns:
            df: Data frame which is a matrix representation
                of the scores for every pair of faces
    '''
    # Generate scores for all pairs of faces
    scores = [ 
        [ float(torch.sum(torch.square(e2_ - e1_))) for e2_ in e2 ] 
        for e1_ in e1 
    ]

    # Convert to data frame for visualization purposes
    df = pd.DataFrame(scores)
    
    # Display the dataframe
    print(df) 

    return df


def get_filtered_imgs(path: str) -> List[Tensor]:
    '''
    Description:
        Helper function to return all images at specified path
        as a batched Tensor (torch)
    
    Args: 
        path: string containing path to dir with images
    
    Returns:
        x: Batched tensor
    '''
    if not os.path.exists(path):
        raise ValueError(f"Path {path} does not exist")

    files = natsorted(glob(os.path.join(path, '*.jpg'))
                  + glob(os.path.join(path, '*.JPG'))
                  + glob(os.path.join(path, '*.png'))
                  + glob(os.path.join(path, '*.PNG')))
    
    if len(files) == 0:
        raise RuntimeError(f"No files at {path}")
    

    for i, f in enumerate(files):
        im = Image.open(f).convert('RGB')
        im = TF.to_tensor(im)
        if i == 0:
            x = im.unsqueeze(0)
        else:
            x = torch.cat( (x, im.unsqueeze(0)) )
    
    return x    
