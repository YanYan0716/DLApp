import torch
LEARNING_RATE = 1e-4
DEVICE = 'cuda' if  torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32
NUM_EPOCHS = 3
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMAGE_DIR = './train_images'
TRAIN_MASK_DIR = './mask_images'
VAL_IMG_DIR = './val_images'
VAL_MASK_DIR = './val_mask/'