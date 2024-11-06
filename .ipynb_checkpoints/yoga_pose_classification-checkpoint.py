from pathlib import Path, PurePath
import os
from sklearn.model_selection import train_test_split
from imutils import paths
import shutil
import random
import matplotlib.pyplot as plt
from PIL import Image
import math
from torchvision import transforms
import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader
import torch
import textwrap
from typing import List, Tuple
import torchvision
import pathlib
import torchinfo


class config:

    # specify the paths to datasets
    DOWNLOAD_DIR = 'YogaPoses'
    TRAIN_DIR = 'data/train'
    VAL_DIR = 'data/val'
    TEST_DIR = 'data/test'

    # set the input height and width
    INPUT_HEIGHT = 224
    INPUT_WIDTH = 224

    # set the input height and width
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def split_image_folder(image_paths: str, folder: str):
    data_path = Path(folder)

    if not data_path.is_dir():
        data_path.mkdir(parents=True, exist_ok=True)

    for path in image_paths:
        full_path = Path(path)
        image_name = full_path.name
        label = full_path.parent.name
        label_folder = data_path / label

        if not label_folder.is_dir():
            label_folder.mkdir(parents=True, exist_ok=True)

        destination = label_folder / image_name
        shutil.copy(path, destination)

