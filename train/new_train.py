import torch
import numpy as np
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.optim as optim
import argparse
from torchvision import transforms as T
from tqdm.auto import tqdm
import time
from data.dataset import create_datasets, create_dataloaders
from utilites.utils import SaveBestModel, save_model, params_count_lite, create_cfg_lists
from models.model import build_rel_scalenet
from models.transformer_based import build_relscaletransformer
from train_utils import train_epoch, validate_epoch

parser = argparse.ArgumentParser()
parser.add_argument('-in_dir', '--megadepth_dir', type=str, default='/media/storagedrive/Megadepth', \
    help='input images folder dir')
parser.add_argument('-cfg', '--cfg', type=str, default='', help='names of folders')
parser.add_argument('-e', '--epochs', type=int, default=50, help='num of epochs to train the model')
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='learning rate for our optimizer')
parser.add_argument('-bs', '--batch_size', type=int, default=64, help='batch size of dataloader')
parser.add_argument('-wd', '--weight_decay', type=float, default=1e-5)

args = vars(parser.parse_args())

BATCH_SIZE = args['batch_size']
DATA_PATH = args['megadepth_dir']
LEARNING_RATE = args['learning_rate']
WEIGHT_DECAY = args['weight_decay']
EPOCHS = args['epochs']
cfg_path = args['cfg']

transform_train = T.Compose([
    T.ToTensor(),
    T.CenterCrop((512, 512)),
    T.Normalize(mean=[0.4604, 0.4661, 0.4107], std=[0.1967, 0.1825, 0.1944]), 
    T.Lambda(lambda x: x.to(device))
])
transform_test = T.Compose([
    T.ToTensor(),
    T.CenterCrop((512, 512)),
    T.Normalize(mean=[0.4604, 0.4661, 0.4107], std=[0.1967, 0.1825, 0.1944]), 
    T.Lambda(lambda x: x.to(device))
])

train_paths, val_paths, test_paths = create_cfg_lists(cfg_path)
#CREATE DATASETS FOR TRAINING
train_dataset, valid_dataset, test_dataset = create_datasets(
    train_paths, val_paths, test_paths, DATA_PATH, transform_train, transform_test
)
#CREATE DATALOADERS FOR TRAINING
train_loader, valid_loader, test_loader = create_dataloaders(
    train_dataset, valid_dataset, test_dataset, batch_size=BATCH_SIZE)
#SHOW ME THE COMPUTATION DEVICE
device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Computation device: {device}\n')

model = build_relscaletransformer(
    pretrained=False,

).to(device)

params_count_lite(model)

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[5, 15, 30], gamma=0.1)
save_best_model = SaveBestModel()

model = model.to(device)
patience = 10
count = 0.
prev_model = None
train_started = time.time()
epoch_start = 0

train_loss = []
valid_loss = []

#START THE TRAINING net, optimizer, train_loader, scale_distr, device

def build_scale_rep(scale_levels=13):

    sub_levels = scale_levels // 2
    scale_base = np.log(np.sqrt(2))
    scale_distr = []

    for i in range(scale_levels):
        scale_distr.append(scale_base * (i - sub_levels))

    return np.asarray(scale_distr)

scale_distr = build_scale_rep()

for epoch in range(EPOCHS):
    print(f'[INFO]: Epoch {epoch+1} of {EPOCHS}')
    train_epoch_loss = train_epoch(
        net=model,
        train_loader=train_loader,
        optimizer=optimizer,
        scale_distr=scale_distr,
        device=device
    )
    #net, val_loader, scale_distr, device
    valid_epoch_loss = validate_epoch(model, valid_loader, scale_distr, device)

    # scheduler.step(valid_epoch_loss)

    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)

    print(f'Training LOSS: {train_epoch_loss:.5f}')
    print(f'Validation LOSS: {valid_epoch_loss:.5f}')

    # save_best_model(valid_epoch_loss, epoch, model, optimizer, criterion, data_path='./weights/model_vgg.pth')
    save_model(epoch, model, optimizer, criterion=None, path_to_save='./weights/model1.pth')
    print('=' * 75)

#SAVE THE TRAINED MODEL WEIGHTS FOR FINAL TIME
save_model(EPOCHS, model, optimizer, criterion=None, path_to_save='./weights/final_model.pth')
print('TRAINING COMPLETE')

# # Start training loop
# for epoch in range(epoch_start, args.n_epoch):

#     # Training one epoch
#     train_loss, diff_scale_train = train_epoch(model, optimizer, train_loader, train_dataset.scale_distr, device)

#     # Validation
#     val_loss, diff_scale_val = validate_epoch(model, valid_loader, train_dataset.scale_distr, device)

#     scheduler.step()

#     # Tensorboard and log save
#     # train_writer.add_scalar('train_loss', train_loss, epoch)
#     # train_writer.add_scalar('val_loss', val_loss, epoch)
#     # train_writer.add_scalar('train_diff_scale', diff_scale_train, epoch)
#     # train_writer.add_scalar('val_diff_scale', diff_scale_val, epoch)

#     # info_log = "\nEpoch: {}. Loss: {:.3f}. Val loss: {:.3f}. Diff scale: {:.3f}. Val diff scale: {:.3f}\n".\
#     #     format(epoch+1, train_loss, val_loss, diff_scale_train, diff_scale_val)
#     # save_to_file(info_log, log_writer)

#     if epoch > args.start_epoch:
#         '''
#         We will be saving only the snapshot which
#         has lowest loss value on the validation set
#         '''
#         cur_name = osp.join(args.snapshots, cur_snapshot, 'epoch_{}.pth'.format(epoch + 1))

#         if prev_model is None:
#             torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, cur_name)
#             prev_model = cur_name
#             best_ratio_val = 10e6
#         else:
#             if diff_scale_val < best_ratio_val:
#                 count = 0.
#                 best_ratio_val = diff_scale_val
#                 os.remove(prev_model)
#                 save_to_file('Saved snapshot: {}\n'.format(cur_name), log_writer)
#                 torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, cur_name)
#                 prev_model = cur_name
#             else:
#                 count += 1

#     # Early stop check
#     if count >= patience:
#         info_log = '\nPatience reached ({}). Best model: {}. Stop Training.'.format(count, prev_model)
#         save_to_file(info_log, log_writer)
#         break

# print(args.seed, 'Training took:', time.time()-train_started, 'seconds')



