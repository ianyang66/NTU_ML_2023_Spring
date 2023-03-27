
import numpy as np
import torch
from torch.utils.data import DataLoader
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
# ## Testing
    model_path = f'./model_{args.model}{args.extra_name}.ckpt'  # the path where the checkpoint will be saved

    # model parameters
    input_dim = 39

    # Create a testing dataset, and load model from the saved checkpoint.
    # load data test_tt_feat, test_total_len
    test_tt_feat, test_total_len = preprocess_data(split='test', feat_dir='./libriphone/feat', phone_path='./libriphone')
    test_set = LibriDataset(test_tt_feat, None)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    # x,y = iter(test_loader).next()
    # print(y.shape)

    # load model
    if args.model == "gru":
        model = GRUClassifier(input_dim=input_dim, hidden_dim=args.hidden_dim, hidden_layers=args.hidden_layers, output_dim=41, dropout = args.dropout, fc_dropout = args.fc_dropout).to(args.device)
    elif args.model == "lstm":
        model = LSTMClassifier(input_dim=input_dim, hidden_dim=args.hidden_dim, hidden_layers=args.hidden_layers, output_dim=41, dropout = args.dropout, fc_dropout = args.fc_dropout).to(args.device)

    model.load_state_dict(torch.load(model_path))


    # Make prediction.
    test_acc = 0.0
    test_lengths = 0
    pred = np.array([], dtype=np.int32)

    ans = []
    test_pred_list = []

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader)):
            features = batch
            features = features.to(args.device)

            outputs = model(features)
            
            # reshape 
            outputs = outputs.view(-1,outputs.size(2))
            ans.append(outputs)

            _, test_pred = torch.max(outputs, 1) # get the index of the class with the highest probability
            test_pred_list.append(test_pred)
            
            pred = np.concatenate((pred, test_pred.cpu().numpy()), axis=0)

    with open(f'prediction{args.extra_name}.csv', 'w') as f:
        f.write('Id,Class\n')
        for i, y in enumerate(pred):
            f.write('{},{}\n'.format(i, y))

def parse_args() -> Namespace:
    parser = ArgumentParser()
    # model
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--hidden_layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--fc_dropout", type=float, default=0.3)
    parser.add_argument("--model", type=str, default="gru")
    parser.add_argument("--extra_name", type=str, default="1")
    # training
    parser.add_argument(
            "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda:0"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)