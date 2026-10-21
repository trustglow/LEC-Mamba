import json
import numpy as np
import random
import argparse
import os 
import warnings
from utils import *
from load_data import load_data
import torch
from model import MyModel
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable as V
from sklearn.model_selection import train_test_split
import model

criterion = torch.nn.CrossEntropyLoss()
windowSize = model.ws 
best_score=0
random_state=345
save=True
load=False
random.seed(69)
output_model = '/tmp/output/model.pth'

test_size=0.95
max_epoch=200

data_path = 'MU'
data_path2 = data_path+'sar'

parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", default=1e-3, type=float,help="learning_rate")
parser.add_argument("--train_bs", default=200, type=int,help="train_bs")

args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten() # [3, 5, 8, 1, 2, ....]
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def save(model, optimizer):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, output_model)

def eval(model, optimizer,criterion,data_loader,save=save):
    print('\033[1;35m----Evaluating----\033[0m')
    model.eval()
    eval_loss, eval_accuracy, nb_eval_steps = 0, 0, 0

    for i,(batch_x,batch_y) in enumerate(tqdm(data_loader)):
        batch_x=[r.to(device) for r in batch_x]
        batch_y=batch_y.to(device)
        with torch.no_grad():
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            logits = logits.detach().cpu().numpy()
            eval_loss += loss.item()
            label_ids = batch_y.cpu().numpy()
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1

    avg_loss = eval_loss / nb_eval_steps 
    avg_accuracy = eval_accuracy / nb_eval_steps
    print(f"Validation Loss: {avg_loss:.4f}, Validation Accuracy: {avg_accuracy:.4f}")  

    global best_score
    if best_score < eval_accuracy / nb_eval_steps:
        best_score = eval_accuracy / nb_eval_steps
        if save:
            save(model, optimizer)
            print('\033[1;31m Better model saved.\033[0m')

    model.train()
    return avg_accuracy
    
def train(model,optimizer,criterion,train_loader,test_loader):
    print('-----------------training------------')
    for epoch in range(max_epoch):
        print('【Epoch:{}】'.format(epoch+1))
        for i,(batch_x,batch_y) in enumerate(tqdm(train_loader)):
            model.train()
            batch_x=[r.to(device) for r in batch_x]
            batch_y=batch_y.to(device)
            logits=model(batch_x)

            loss=criterion(logits,batch_y)
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        eval(model, optimizer,criterion,test_loader)

import shutil  
import torch.nn.init as init  

if __name__ == '__main__':
    print('start')

    data, gt = load_data(data_path)  
    data = Standardize_data(data)

    data2, gt = load_data(data_path2)  
    data2 = Standardize_data(data2)

    full_dataset = ComplexImageCubeDataset(data,data2, gt, windowSize=windowSize)
    del data, gt,data2

    all_indices = np.arange(0, len(full_dataset))


    labels_for_stratify = full_dataset.labels[all_indices]


    train_idx, test_idx = train_test_split(
    all_indices,
    test_size=test_size,
    random_state=random_state,
    stratify=labels_for_stratify
    )
    
    del labels_for_stratify,all_indices
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
    test_dataset = torch.utils.data.Subset(full_dataset, test_idx)
    
    del train_idx,test_idx,full_dataset
    
    train_loader = create_dataloader(train_dataset, args.train_bs, shuffle=True)
    test_loader = create_dataloader(test_dataset, args.train_bs, shuffle=False)
    
    del train_dataset,test_dataset

    model=MyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,weight_decay=0.01)
    
    warnings.filterwarnings("ignore", message="Applied workaround for CuDNN issue, install nvrtc.so")

    if not os.path.exists('/tmp/output'):  
        os.makedirs('/tmp/output')  
    if not os.path.exists('pretrain'):  
        os.makedirs('pretrain')  
    model.to(device)
    
    if load:
        checkpoint = torch.load(output_model)  
        model.load_state_dict(checkpoint['model_state_dict'])  
        optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 
        print('Pretrained model loaded.')

    eval(model, optimizer,criterion,test_loader,save=False)
    train(model,optimizer,criterion,train_loader,test_loader)
    print(f'Best score:{best_score:.4f}')