import torch
import os
import shutil
import cv2
import torch.nn as nn
import numpy as np
import random
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import torch.nn.init as init

class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(self, best_valid_loss=float('inf')):
        self.best_valid_loss = best_valid_loss
        
    def __call__(self, current_valid_loss, epoch, model, optimizer, criterion, data_path='weights/best_model.pth'):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, data_path)

def save_model(epochs, model, optimizer, criterion, path_to_save='./weights/model_vgg_just.pth'):
    """
    Function to save the trained model to disk.
    """
    print(f"Saving final model...")
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, path_to_save)


def _initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)


def params_count(model):

    total_params = sum(p.numel() for p in model.parameters())
    total_bb_params = sum(p.numel() for p in model.bb.parameters())
    total_dc_params = sum(p.numel() for p in model.dc.parameters())
    total_lin_params = sum(p.numel() for p in model.lin.parameters())
    print(f"total parameters: ", {total_params})
    print(f"total front parameters: ", {total_bb_params})
    print(f"total back parameters: ", {total_dc_params})
    print(f"total regr parameters: ", {total_lin_params})



    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    train_bb_params = sum(p.numel() for p in model.bb.parameters() if p.requires_grad)
    train_dc_params = sum(p.numel() for p in model.dc.parameters() if p.requires_grad)
    train_lin_params = sum(p.numel() for p in model.lin.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.\n")
    print(f"train front parameters: ", {train_bb_params})
    print(f"train back parameters: ", {train_dc_params})
    print(f"train regr parameters: ", {train_lin_params})


def params_count_lite(model):
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params}')
    print(f'Trainable parameters: {total_trainable_params}')

def center_crop(img, size):
    """
    Get the center crop of the input image
    Args:
        img: input image [BxCxHxW]
        size: size of the center crop (tuple)
    Output:
        img_pad: center crop
        x, y: coordinates of the crop
    """

    if not isinstance(size, tuple):
        size = (size, size)

    img = img.copy()
    w, h = img.shape[1::-1]

    pad_w = 0
    pad_h = 0
    if w < size[0]:
        pad_w = np.uint16((size[0] - w) / 2)
    if h < size[1]:
        pad_h = np.uint16((size[1] - h) / 2)
    img_pad = cv2.copyMakeBorder(img, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    w, h = img_pad.shape[1::-1]

    x1 = w // 2 - size[0] // 2
    y1 = h // 2 - size[1] // 2

    img_pad = img_pad[y1:y1 + size[1], x1:x1 + size[0], :]

    return img_pad, x1, y1


def create_cfg_lists(cfg_path: str, val_size=2, test_size=1, random_seed=12):
    file1 = open(cfg_path, 'r')
    a = [line.split(', ') for line in file1.readlines()]
    cfg = a[0][:-1]
    train_size = 183 - val_size - test_size
    train_set, others = train_test_split(cfg, test_size=val_size+test_size, random_state=random_seed)
    val_set, test_set = train_test_split(others, test_size=test_size, random_state=random_seed)

    return train_set, val_set, test_set


def criterion_KL_divergence(dist1, dist2, beta=100):
    loss = beta * F.kl_div(F.log_softmax(dist1, 1), dist2, reduction='none').mean()
    return loss




