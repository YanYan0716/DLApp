import torch
import torch.nn as nn
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
import albumentations as A
from tqdm import tqdm

from model import UNet
from utils import load_checkpoint
from utils import save_checkpoint
from utils import get_loaders
from utils import check_accuracy
from utils import save_predictions_as_imgs
import config


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=config.DEVICE)
        targets = targets.float().unqueeze(1).to(device=config.DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
    train_transform = A.Compose([
        A.Resize(height=config.IMAGE_HEIGHT, width=config.IMAGE_WIDTH),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2,
    ])

    val_transform = A.Compose([
        A.Resize(height=config.IMAGE_HEIGHT, width=config.IMAGE_WIDTH),
        A.normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2,
    ])

    model = UNet(in_channels=3, out_channels=1).to(config.DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        config.TRAIN_IMAGE_DIR,
        config.TRAIN_MASK_DIR,
        config.VAL_IMG_DIR,
        config.VAL_MASK_DIR,
        config.BATCH_SIZE,
        train_transform,
        val_transform,
    )

    if config.LOAD_MODEL:
        load_checkpoint(torch.load('my_checkpoint.pth.tar'), model)

    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(config.NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # check acc
        check_accuracy(val_loader, model, device=config.DEVICE)

        # print some examples to a folder
        save_predictions_as_imgs(val_loader, model, folder='saved_images', device=config.DEVICE)


if __name__=='__main__':
    pass