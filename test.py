import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import argparse
from torchvision import transforms as T
from tqdm.auto import tqdm

from dataset import MegaDepthDataset
from transformer_based import build_relscaletransformer

parser = argparse.ArgumentParser()
parser.add_argument('-in_dir', '--megadepth_dir', type=str, default='/media/storagedrive/Megadepth', \
    help='input images folder dir')
parser.add_argument('-name', '--folder_name', type=str, default='0214', help='name or index of folder')
parser.add_argument('-bs', '--batch_size', type=int, default=64, help='batch size of dataloader')

args = vars(parser.parse_args())

BATCH_SIZE = args['batch_size']
DATA_PATH = args['megadepth_dir']
FOLDER_NAME = args['folder_name']

transform_test = T.Compose([
    T.ToTensor(),
    T.CenterCrop((512, 512)),
    T.Normalize(mean=[0.4604, 0.4661, 0.4107], std=[0.1967, 0.1825, 0.1944]), 
    T.Lambda(lambda x: x.to(device))
])


test_set = MegaDepthDataset(FOLDER_NAME, DATA_PATH, transform_test)
testloader = DataLoader(test_set, batch_size=BATCH_SIZE)
#SHOW ME THE COMPUTATION DEVICE
device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Computation device: {device}\n')
criterion = nn.MSELoss()
criterion1 = nn.L1Loss()

model = build_relscaletransformer(pretrained=True).to(device)

def test(model, testloader, criterion):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_loss1 = 0.0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            
            images1, images2, scales = data
            scales = scales.to(device)
            # forward pass
            outputs = model(images1, images2)
            outputs = outputs.view(-1)
            # calculate the loss
            loss = criterion(outputs, scales)
            valid_running_loss += loss.item()
            loss1 = criterion1(outputs,scales)
            valid_running_loss1 += loss1.item()
            if counter % 10 == 0:
                print(f'outputs|labels\n {torch.stack((outputs,scales),1)}')
                print(f'MSE Loss: {valid_running_loss/counter:.8f} Batch_Loss: {loss.item()}\n')
                print(f'L1Loss: {valid_running_loss1/counter:.8f} Batch_Loss: {loss.item()}\n')
                with open('./logs/test_logs.txt', 'a') as f:
                    f.write(f'Loss: {valid_running_loss/counter:.8f} Batch_Loss: {loss.item()}\n')
        
    # loss and accuracy for the complete epoch
    epoch_loss = valid_running_loss / counter
    epoch_loss1 = valid_running_loss1 / counter
    return epoch_loss, epoch_loss1

l, l1 = test(model, testloader, criterion)

print(f'RESULT of Testing: {l} MSE Loss for folder {FOLDER_NAME}')
print(f'RESULT of Testing: {l1} L1Loss for folder {FOLDER_NAME}')
