import torch
import torch.nn as nn
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
import albumentations as A
from tqdm import tqdm

from model import UNet
import config


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=config.DEVICE)
        targets = targets.float().unqueeze(1).to(device=config.DEVICE)

        # forward



def main():
    pass



if __name__=='__main__':
    pass