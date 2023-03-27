
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import gc
from utils import *
from model import *
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
from preprocess_v2 import *
from dataset import LibriDataset
import gc

torch.cuda.empty_cache()

def main(args):
    model_path = f'./model_{args.model}{args.extra_name}.ckpt'
    # Hyper-parameters
    # The way I changed all arguments to ArgumentParser.
    input_dim = 39 #* concat_nframes # the input dim of the model, you should not change the value
    
    # ## Prepare dataset and model
    # preprocess data
    train_feat, train_label, train_len = preprocess_data(split='train', feat_dir='./libriphone/feat', phone_path='./libriphone', train_ratio=args.train_ratio)
    val_feat, val_label, val_total_len = preprocess_data(split='val', feat_dir='./libriphone/feat', phone_path='./libriphone', train_ratio=args.train_ratio)

    train_set = LibriDataset(train_feat, train_label)
    del train_feat, train_label
    gc.collect()
    val_set = LibriDataset(val_feat, val_label)
    del val_feat, val_label
    gc.collect()
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,num_workers=4, collate_fn=collate_fn)

    # ## Training
    # fix random seed
    same_seeds(args.seed)

    # create model, define a loss function, and optimizer
    if args.model == "gru":
        model = GRUClassifier(input_dim=input_dim, hidden_dim=args.hidden_dim, hidden_layers=args.hidden_layers, output_dim=41, dropout = args.dropout, fc_dropout = args.fc_dropout).to(args.device)
    elif args.model == "lstm":
        model = LSTMClassifier(input_dim=input_dim, hidden_dim=args.hidden_dim, hidden_layers=args.hidden_layers, output_dim=41, dropout = args.dropout, fc_dropout = args.fc_dropout).to(args.device)
    if args.loss == "FocalLoss":
        criterion = FocalLoss()
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=42) 
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # model.load_state_dict(torch.load(model_path))
    
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
            
            # reshape 
            outputs = outputs.view(-1,outputs.size(2))
            labels = labels.view(-1)
            loss = criterion(outputs, labels)
            loss = loss/args.accum_steps
            loss.backward()
            if i % args.accum_steps == 0 or i == len(train_loader) - 1:
                optimizer.step() 
            _, train_pred = torch.max(outputs, 1) # get the index of the class with the highest probability
            train_acc += (train_pred.detach() == labels.detach()).sum().item()
            train_loss += loss.item()
        
        # validation
        if len(val_set) > 0:
            model.eval() # set the model to evaluation mode
            with torch.no_grad():
                for i, batch in enumerate(tqdm(val_loader)):
                    features, labels = batch
                    features = features.to(args.device)
                    labels = labels.to(args.device)
                    outputs = model(features)
                    
                    # reshape 
                    outputs = outputs.view(-1,outputs.size(2))
                    labels = labels.view(-1)
                    
                    loss = criterion(outputs, labels) 
                    
                    _, val_pred = torch.max(outputs, 1) 
                    val_acc += (val_pred.cpu() == labels.cpu()).sum().item() # get the index of the class with the highest probability
                    val_loss += loss.item()

                print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(
                    epoch + 1, args.num_epoch, train_acc/train_len, train_loss/len(train_loader), val_acc/val_total_len, val_loss/len(val_loader)
                ))

                # if the model improves, save a checkpoint at this epoch
                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save(model.state_dict(), model_path)
                    print('saving model with acc {:.3f}'.format(best_acc/val_total_len))
        else:
            print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f}'.format(
                epoch + 1, args.num_epoch, train_acc/train_len, train_loss/len(train_loader)
            ))

    # if not validating, save the last epoch
    if len(val_set) == 0:
        torch.save(model.state_dict(), model_path)
        print('saving model at last epoch')

def parse_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument("--train_ratio", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=121314)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--learning_rate", type=float, default=5e-4)

    # model
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--hidden_layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--fc_dropout", type=float, default=0.3)
    parser.add_argument("--model", type=str, default="gru")
    # training
    parser.add_argument(
            "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda:0"
    )
    parser.add_argument("--num_epoch", type=int, default=300)
    parser.add_argument("--loss", type=str, default="CrossEntropyLoss")
    parser.add_argument("--accum_steps", type=int, default=1)
    parser.add_argument("--extra_name", type=str, default="1")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
