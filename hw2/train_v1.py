"""
**Fixes random number generator seeds for reproducibility.**
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import gc
from utils import *
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
from preprocess_v1 import *
from dataset import LibriDataset
from model import Classifier
torch.cuda.empty_cache()

"""# Hyper-parameters"""

"""# Dataloader"""

def main(args):
    model_path = f'./model_{args.model}.ckpt'
    # Hyper-parameters
    # The way I changed all arguments to ArgumentParser.
    # TODO: change the value of "hidden_layers" or "hidden_dim" for medium baseline
    input_dim = 39 * args.concat_nframes  # the input dim of the model, you should not change the value
    
    # preprocess data
    train_X, train_y = preprocess_data(split='train', feat_dir='./libriphone/feat', phone_path='./libriphone', concat_nframes=args.concat_nframes, train_ratio=args.train_ratio)

    # get dataset
    train_set = LibriDataset(train_X, train_y)
    del train_X, train_y
    gc.collect()
    # get dataloader
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    
    # preprocess data
    val_X, val_y = preprocess_data(split='val', feat_dir='./libriphone/feat', phone_path='./libriphone', concat_nframes=args.concat_nframes, train_ratio=args.train_ratio)
    # get dataset
    val_set = LibriDataset(val_X, val_y)
    del val_X, val_y
    gc.collect()
    # get dataloader    
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, pin_memory=True)
 
    same_seeds(args.seed)

    # Training
    # create model, define a loss function, and optimizer
    if args.model=="nn":
        model = Classifier(input_dim=input_dim, hidden_layers=args.hidden_layers, hidden_dim=args.hidden_dim,output_dim=41).to(args.device)
    criterion = nn.CrossEntropyLoss() 
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    best_acc = 0.0
    for epoch in range(args.num_epoch):
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0
        
        # training
        model.train() # set the model to training mode
        for i, batch in enumerate(tqdm(train_loader)):
            features, labels = batch
            features = features.to(args.device)
            labels = labels.to(args.device)
            
            optimizer.zero_grad() 
            outputs = model(features) 
            
            loss = criterion(outputs, labels)
            loss.backward() 
            optimizer.step() 
            
            _, train_pred = torch.max(outputs, 1) # get the index of the class with the highest probability
            train_acc += (train_pred.detach() == labels.detach()).sum().item()
            train_loss += loss.item()
        
        # validation
        model.eval() # set the model to evaluation mode
        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_loader)):
                features, labels = batch
                features = features.to(args.device)
                labels = labels.to(args.device)
                outputs = model(features)
                
                loss = criterion(outputs, labels) 
                
                _, val_pred = torch.max(outputs, 1) 
                val_acc += (val_pred.cpu() == labels.cpu()).sum().item() # get the index of the class with the highest probability
                val_loss += loss.item()

        print(f'[{epoch+1:03d}/{args.num_epoch:03d}] Train Acc: {train_acc/len(train_set):3.5f} Loss: {train_loss/len(train_loader):3.5f} | Val Acc: {val_acc/len(val_set):3.5f} loss: {val_loss/len(val_loader):3.5f}')

        # if the model improves, save a checkpoint at this epoch
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_path)
            #files.download(model_path)
            print(f'saving model with acc {best_acc/len(val_set):.5f}')

    del train_set, val_set
    del train_loader, val_loader
    gc.collect()

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--concat_nframes", type=int, default=47)
    parser.add_argument("--train_ratio", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=121314)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--learning_rate", type=float, default=5e-4)

    # model
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--hidden_layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--model", type=str, default="nn")
    # training
    parser.add_argument(
            "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda:0"
    )
    parser.add_argument("--num_epoch", type=int, default=300)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)