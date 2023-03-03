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

# Facenet Imports
from facenet_pytorch import *

# Utility Functions
from utils import *


def main():
    # Initialize vid reader
    vid_path = 'facenet_pytorch/examples/video.mp4'
    vid = cv2.VideoCapture(vid_path)

    # Initialize vid writer
    W, H = int(vid.get(3)), int(vid.get(4))
    out_path = 'results/out.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 60.0
    out = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
    
    # Open and parse frames from filestream
    if not vid.isOpened():
        print('Error opening file stream')
    
    # Detect and initialize device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using Device: {}".format(device))
    
    # Initialize face extraction and Inception models
    clip_size = 224
    mtcnn = MTCNN( clip_size, keep_all=True, device=device )
    model = InceptionResnetV1(pretrained='vggface2', device=device)
    model.eval()

    # Read frames from video and prepare
    # faces for InceptionResnet model
    r = 1
    faces_aligned = []
    start = time.time()
    while vid.isOpened() and r:
        r, frame = vid.read()
        if not r:
            break
        
        img = torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        faces = mtcnn(img)
        faces_aligned.append(faces)

        # TODO: remove the write from this loop
        out.write(frame)

    # Release reader/writer objects and close all frames
    vid.release()
    out.release()
    cv2.destroyAllWindows()

    preprocessing_end = time.time()

    # Pass the aligned faces to the InceptionResnet model
    # for generating embeddings and do some book keeping
    test_frame = faces_aligned[0].to(device)
    save_aligned_faces(test_frame)

    inf_start = time.time()
    embeddings = model(test_frame).detach().cpu()
    end = time.time()

    # Generate a similarity matrix of all faces detected
    scores = [ 
        [ float(torch.sum(torch.square(e2 - e1))) for e2 in embeddings ] 
        for e1 in embeddings 
    ]
    
    # Display Results
    df = pd.DataFrame(scores)
    print(df)
    print("Preprocesing Time: {:.2f} s".format(preprocessing_end - start))
    print("InceptionResnet Inference time: {:.2f} s".format(end - inf_start))


if __name__ == '__main__':
    main()
