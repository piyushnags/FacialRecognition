'''
main module for using the full
facial recognition pipeline.
Please see the GitHub repo for
more info about running the code
and updates on the AutoEncoder:
https://github.com/piyushnags/FacialRecognition
'''

# Buit-in Imports
import os, time, sys

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
from train_utils import InceptionDataset
from eval_utils import *


def main(args: Any):
    '''
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

    # Initialize working dir
    clear_results()

    # Initialize vid reader
    vid_path = args.vid_path
    vid = cv2.VideoCapture(vid_path)

    # Initialize vid writer
    W, H = int(vid.get(3)), int(vid.get(4))

    if not os.path.exists('results/'):
        os.makedirs('results')

    out_path = 'results/out.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30.0
    out = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
    
    # Open and parse frames from filestream
    if not vid.isOpened():
        print('Error opening file stream')
    
    # Detect and initialize device
    if args.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    print("Using Device: {}".format(device))
    
    # Initialize face extraction and Inception models
    # Clip to 256 for the SUNet transformer
    clip_size = 256
    mtcnn = MTCNN( clip_size, keep_all=True, device=device )
    model = InceptionResnetV1(pretrained='vggface2', device=device)
    inception_weights = torch.load('weights/inception.pth', map_location=device)
    model.load_state_dict(inception_weights)
    model.eval()

    # Initialize additive noise as a torch 
    # transform (scale the noise accordingly since 
    # images are whitened before using the MTCNN)
    noisify = T.Compose([
        AddNoise(args.noise_var, args.noise_mean)
    ])

    # Read frames from video and prepare
    # faces for InceptionResnet model
    r = 1
    faces_aligned = []
    noisy_faces = []
    start = time.time()
    curr = 1
    while vid.isOpened() and r:
        if curr > args.max_frames:
            break
        curr += 1

        r, frame = vid.read()
        if not r:
            break
        
        img = torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        faces = mtcnn(img)
        faces_aligned.append(faces)

        # compute noisy images using a deep copy of the tensor
        noisy_faces.append( noisify(faces.clone()) )

        # NOTE: logging the content actually processed into an out.mp4 file
        out.write(frame)

    # Release reader/writer objects and close all frames
    vid.release()
    out.release()
    cv2.destroyAllWindows()

    preprocessing_end = time.time()

    frame_idx = 20
    # Save the clean faces
    clean_frame = faces_aligned[frame_idx][:args.max_faces]
    save_faces(clean_frame, 'aligned_faces')
    
    # Save the noisy faces
    noisy_frame = noisy_faces[frame_idx][:args.max_faces]
    save_faces(noisy_frame, 'noisy_faces')

    # Pass the aligned and noisy faces to the InceptionResnet model
    # for generating embeddings and do some book keeping
    inf_start = time.time()
    clean_frame, noisy_frame = clean_frame.to(device), noisy_frame.to(device)

    with torch.no_grad():
        embeddings_clean = model(clean_frame).detach().cpu()
        embeddings_noisy = model(noisy_frame).detach().cpu()

    end = time.time()

    # Generate a similarity matrix of all faces detected
    # Note that the same pairs of faces are being compared for
    # the noisy and clean images
    print("Comparing Original images and Noisy images")
    _ = evaluate_embeddings(embeddings_clean, embeddings_noisy)

    # Logging generic stats about execution time
    print("Preprocesing Time: {:.2f} s".format(preprocessing_end - start))
    print("InceptionResnet Inference time: {:.2f} s".format(end - inf_start))

    # Get options for the model and generate clean images
    # and do some profiling
    transformer_start = time.time()
    options = get_options(args.yaml_path)
    clean_images(args.input_dir, args.result_dir, args.weights, options, device)
    print(f"SUNet inference time (cleaning images): {time.time()-transformer_start:.2f} s")

    # Generate similarity matrix for filtered images and clean images
    filtered_imgs = get_filtered_imgs(args.result_dir, disable_blur=args.disable_blur)
    filtered_imgs = filtered_imgs.to(device)
    with torch.no_grad():
        embeddings_filtered = model(filtered_imgs).detach().cpu()

    print("Comparing Original images and Filtered images")
    _ = evaluate_embeddings(embeddings_clean, embeddings_filtered)



if __name__ == '__main__':
    args = parse_options()
    
    # Evaluate model performance and compute SSIM of restoration
    if args.eval_model:
        if args.device == 'cuda':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device('cpu')
        model = InceptionResnetV1(pretrained='vggface2', device=device)
        dataset = InceptionDataset(args.root1, args.root2)
        evaluate_similarity(model, dataset, args.batch_size, args.num_workers, device)
        sys.exit(0)

    # Call main demo script
    main(args)
