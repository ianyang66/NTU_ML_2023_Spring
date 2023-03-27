import numpy as np
import torch
from torch.utils.data import DataLoader
from utils import *
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
from preprocess_v1 import *
from dataset import LibriDataset
from model import Classifier

torch.cuda.empty_cache()
"""# Dataloader"""

def main(args):
    # training parameters
    model_path = f'./model_{args.model}.ckpt'  # the path where the checkpoint will be saved

    # model parameters
    # TODO: change the value of "hidden_layers" or "hidden_dim" for medium baseline
    input_dim = 39 * args.concat_nframes  # the input dim of the model, you should not change the value
    """# Testing
    Create a testing dataset, and load model from the saved checkpoint.
    """

    # load data
    test_X = preprocess_data(split='test', feat_dir='./libriphone/feat', phone_path='./libriphone', concat_nframes=args.concat_nframes)
    test_set = LibriDataset(test_X, None)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # load model
    model = Classifier(input_dim=input_dim, hidden_layers=args.hidden_layers, hidden_dim=args.hidden_dim,output_dim=41).to(args.device)
    model.load_state_dict(torch.load(model_path))

    """Make prediction."""

    pred = np.array([], dtype=np.int32)

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader)):
            features = batch
            features = features.to(args.device)

            outputs = model(features)

            _, test_pred = torch.max(outputs, 1) # get the index of the class with the highest probability
            pred = np.concatenate((pred, test_pred.cpu().numpy()), axis=0)

    """Write prediction to a CSV file.

    After finish running this block, download the file `prediction.csv` from the files section on the left-hand side and submit it to Kaggle.
    """

    with open('prediction.csv', 'w') as f:
        f.write('Id,Class\n')
        for i, y in enumerate(pred):
            f.write('{},{}\n'.format(i, y))


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
