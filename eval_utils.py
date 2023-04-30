# Built-in Imports
from typing import Any, Tuple, List
from tqdm import tqdm

# Math and Visualization Imports
from pytorch_msssim import ssim
import matplotlib.pyplot as plt

# PyTorch Imports
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset



def evaluate_similarity(model: nn.Module, dataset: Dataset, batch_size: int, num_workers: int, device: Any):
    model.eval()
    loss_fn = nn.MSELoss()

    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    running_ssim = []
    running_mse = []
    for i, (img1, img2) in enumerate(tqdm(loader)):
        if i == 0:
            plt.imshow(img1[0].permute(1,2,0))
            plt.imsave('test.png', img1[0].permute(1,2,0))
        img1, img2 = img1.to(device), img2.to(device)
        print(img1.max())
        print(img1.min())
        print(img1.shape)

        score = ssim(img1, img2, data_range=1, size_average=True)
        # score = torch.mean(score)
        running_ssim.append(score.item())

        with torch.no_grad():
            out1 = model(img1)
            out2 = model(img2)
            loss = loss_fn(out1, out2)
            running_mse.append(loss.item())


    print(f"DEBUG: (SSIM) {running_ssim[0]},  {running_ssim[2]}")
    print(f"DEBUG: (MSE) {running_mse[0]},  {running_mse[2]}")
    avg_ssim = sum(running_ssim)/len(running_ssim)
    avg_mse = sum(running_mse)/len(running_mse)
    print(f"Average SSIM for Datasets is {avg_ssim:.5f} ") 
    print(f"Average MSE for Datasets is {avg_mse:.5f}")