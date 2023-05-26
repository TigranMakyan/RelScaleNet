import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import argparse
from torchvision import transforms as T
from tqdm.auto import tqdm
from torch.utils.data import Subset

from data.google_dataset import create_google_dataloaders, GoogleDataset_entire_image
from utilites.utils import SaveBestModel, save_model, params_count_lite
# from models.model import build_rel_scalenet
from models.transformer_based import build_relscaletransformer
#torch.multiprocessing.set_start_method('spawn')
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='/media/storagedrive/map/', \
    help='input images folder dir')
parser.add_argument('-e', '--epochs', type=int, default=50, help='num of epochs to train the model')
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='learning rate for our optimizer')
parser.add_argument('-bs', '--batch_size', type=int, default=64, help='batch size of dataloader')

args = vars(parser.parse_args())

DATA_PATH = args['data_path']
BATCH_SIZE = args['batch_size']
LEARNING_RATE = args['learning_rate']
EPOCHS = args['epochs']
device = ('cuda' if torch.cuda.is_available() else 'cpu')

transform_train = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.4604, 0.4661, 0.4107], std=[0.1967, 0.1825, 0.1944]), 
    #T.Lambda(lambda x: x.to(device))
])

total_dataset = GoogleDataset_entire_image(transform=transform_train, data_path=DATA_PATH)
#CREATE DATASETS FOR TRAINING

train_idx, validation_idx = train_test_split(np.arange(len(total_dataset)),
                                             test_size=0.1,
                                             random_state=2023,
                                             shuffle=True)

# Subset dataset for train and val
train_dataset = Subset(total_dataset, train_idx)
valid_dataset = Subset(total_dataset, validation_idx)

#CREATE DATALOADERS FOR TRAINING
train_loader, valid_loader = create_google_dataloaders(
    train_dataset, valid_dataset, batch_size=BATCH_SIZE)
#SHOW ME THE COMPUTATION DEVICE
print(f'Computation device: {device}\n')

model = build_relscaletransformer(pretrained=False).to(device)
#_initialize_weights(model)
#model = build_rel_scalenet(pretrained=False, freeze_bb=True, freeze=False).to(device)
print(model)
checkpoint = torch.load('/home/powerx/computer_vision/RelScale/weights/google_data_model_10_epoch.pth')
model.load_state_dict(checkpoint['model_state_dict'])
params_count_lite(model)

optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=0)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 25], gamma=0.3)
criterion = nn.MSELoss()
save_best_model = SaveBestModel()

def train_epoch(model, trainloader, optimizer, criterion, epoch):
    model.train()
    print('[INFO]: TRAINING IS RUNNING')
    train_running_loss = 0.0
    counter = 0
    shochik = -1
    with tqdm(enumerate(trainloader),total=len(trainloader)) as tepoch:
        for i, data in tepoch:
            tepoch.set_description(f"iter {i}/{len(trainloader)}")
            shochik+=1
            counter += 1
            images1, images2, scales = data
            images1 = images1.to(device)
            images2 = images2.to(device)
            # scales = torch.tensor(scales, device=torch.device(device), dtype=torch.float)
            scales = torch.tensor(scales, device=torch.device(device))

            optimizer.zero_grad()
            outputs = model(images1, images2)
            outputs = outputs.view(-1).to(torch.float64)
            loss = criterion(outputs, scales)
            l2_lambda = 0.000000001
            l2_norm = sum(p.abs().sum() for p in model.parameters())
        
            loss = loss + l2_lambda * l2_norm
            train_running_loss += loss.item()
            loss.backward()
            optimizer.step()
            if counter % 50 == 0:
                tepoch.set_postfix(loss=train_running_loss/counter)
            if shochik % 100 == 0:
                print(outputs)
                with open('./logs/train_logs1.txt', 'a') as f:
                    f.write(f'Epoch: {epoch+1} Iter: {shochik/len(trainloader)*100:.0f}% Loss: \
                        {train_running_loss/counter:.8f} Batch_Loss: {loss.item():.8f}\n')
    
    epoch_loss = train_running_loss / counter
    return epoch_loss


def validate(model, testloader, criterion,epoch):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            
            images1, images2, scales = data
            images1 = images1.to(device)
            images2 = images2.to(device)
            scales = torch.tensor(scales, device=torch.device(device), dtype=torch.float32)
           # scales = scales.to(device)
            # forward pass
            outputs = model(images1, images2)
            outputs = outputs.view(-1).to(torch.float32)
            # calculate the loss
            loss = criterion(outputs, scales)
            valid_running_loss += loss.item()
            if counter % 5 == 0:
                print(f'outputs|labels\n {torch.stack((outputs,scales),1)}')
                with open('./logs/val_logs.txt', 'a') as f:
                    f.write(f'Epoch: {epoch} Loss: {valid_running_loss/counter:.8f} Batch_Loss: {loss.item()}\n')
            # calculate the accuracy
        
    # loss and accuracy for the complete epoch
    epoch_loss = valid_running_loss / counter
    return epoch_loss

train_loss = []
valid_loss = []

#START THE TRAINING

for epoch in range(EPOCHS):
    print(f'[INFO]: Epoch {epoch+1} of {EPOCHS}')
    train_epoch_loss = train_epoch(
        model=model,
        trainloader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        epoch=epoch
    )
    valid_epoch_loss = validate(model, valid_loader, criterion,epoch=epoch)
    scheduler.step()
    # scheduler.step(valid_epoch_loss)

    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)

    print(f'Training LOSS: {train_epoch_loss:.5f}')
    print(f'Validation LOSS: {valid_epoch_loss:.5f}')

    # save_best_model(valid_epoch_loss, epoch, model, optimizer, criterion, data_path='./weights/model_vgg.pth')
    save_model(epoch, model, optimizer, criterion, path_to_save=f'./models/google_new_data_model_{epoch}_epoch.pth')
    print('=' * 75)

# #SAVE THE TRAINED MODEL WEIGHTS FOR FINAL TIME
# save_model(EPOCHS, model, optimizer, criterion, path_to_save='./models/final_model_tr.pth')
# print('TRAINING COMPLETE')


