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
import torchvision
import torchvision.transforms as T

# Facenet Imports
from facenet_pytorch import *

# Utility Functions
from utils import *


def main():
    '''
    TODO: Add support for argparse/configparse

    Description:
    main function that demonstrates the performance of the 
    Inception Resnet network on video footage before and after
    adding corruption, and on the cleaned up footage using the
    transformer.

    Demo video can be found as out.mp4 in results

    Args:
        None
    
    Returns:
        None
    '''

    # Initialize vid reader
    vid_path = 'facenet_pytorch/examples/video.mp4'
    vid = cv2.VideoCapture(vid_path)

    # Initialize vid writer
    W, H = int(vid.get(3)), int(vid.get(4))
    out_path = 'results/out.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 10.0
    out = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
    
    # Open and parse frames from filestream
    if not vid.isOpened():
        print('Error opening file stream')
    
    # Detect and initialize device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using Device: {}".format(device))
    
    # Initialize face extraction and Inception models
    # Clip to 256 for the SUNet transformer
    clip_size = 256
    mtcnn = MTCNN( clip_size, keep_all=True, device=device )
    model = InceptionResnetV1(pretrained='vggface2', device=device)
    model.eval()

    # Initialize additive noise as a torch 
    # transform (scale the noise accordingly since 
    # images are whitened before using the MTCNN)
    noisify = T.Compose([
        AddNoise(1e-2, 1e-2)
    ])

    # Read frames from video and prepare
    # faces for InceptionResnet model
    r = 1
    faces_aligned = []
    noisy_faces = []
    start = time.time()
    while vid.isOpened() and r:
        r, frame = vid.read()
        if not r:
            break
        
        img = torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        faces = mtcnn(img)
        faces_aligned.append(faces)

        # compute noisy images using a deep copy of the tensor
        noisy_faces.append( noisify(faces.clone()) )

        # TODO: remove the write from this loop
        out.write(frame)

    # Release reader/writer objects and close all frames
    vid.release()
    out.release()
    cv2.destroyAllWindows()

    preprocessing_end = time.time()

    # Save the clean faces
    clean_frame = faces_aligned[0]
    save_faces(clean_frame, 'aligned_faces')
    
    # Save the noisy faces
    noisy_frame = noisy_faces[0]
    save_faces(noisy_frame, 'noisy_faces')

    # Pass the aligned and noisy faces to the InceptionResnet model
    # for generating embeddings and do some book keeping
    inf_start = time.time()
    clean_frame, noisy_frame = clean_frame.to(device), noisy_frame.to(device)
    embeddings_clean = model(clean_frame).detach().cpu()
    embeddings_noisy = model(noisy_frame).detach().cpu()
    end = time.time()

    # Generate a similarity matrix of all faces detected
    # Note that the same pairs of faces are being compared for
    # the noisy and clean images
    evaluate_embeddings(embeddings_clean, embeddings_noisy)

    # Logging generic stats about execution time
    print("Preprocesing Time: {:.2f} s".format(preprocessing_end - start))
    print("InceptionResnet Inference time: {:.2f} s".format(end - inf_start))



if __name__ == '__main__':
    args = parse_sunet()
    
    # Collect aligned faces
    main()

    # Determine device for transformer inference
    if args.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    
    # Get options for the model and generate clean images
    options = get_options(args.yaml_path)
    clean_images(args.input_dir, args.result_dir, args.weights, options, device)

