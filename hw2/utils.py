import numpy as np
import random
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reference: https://wandb.ai/capecape/classification-techniques/reports/Classification-Loss-Functions-Comparing-SoftMax-Cross-Entropy-and-More--VmlldzoxODEwNTM5
class FocalLoss(nn.Module):
    "Focal loss implemented using F.cross_entropy"
    def __init__(self, gamma: float = 2.0, weight=None, reduction: str = 'mean') -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inp: torch.Tensor, targ: torch.Tensor):
        ce_loss = F.cross_entropy(inp, targ, weight=self.weight, reduction="none")
        p_t = torch.exp(-ce_loss)
        loss = (1 - p_t)**self.gamma * ce_loss
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss

def same_seeds(seed):
    random.seed(seed) 
    np.random.seed(seed)  
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# Reference: https://allenlu2007.wordpress.com/2019/02/07/%E6%B7%B1%E5%BA%A6%E5%AD%B8%E7%BF%92%E7%9A%84-pytorch-load-dataset/
def collate_fn(data):
    # feat, label = [item[0] for item in data], [item[1] for item in data]
    feat, label = zip(*data)
    feat = sorted(feat, key=len, reverse=True)
    label = sorted(label, key=len, reverse=True)
    # feat.sort(key=lambda x: len(x), reverse=True)
    # label.sort(key=lambda x: len(x), reverse=True)
    pad_feat = pad_sequence(feat, batch_first=True)    
    pad_label = pad_sequence(label, batch_first=True, padding_value=42)   
    packed_feat = pack_padded_sequence(pad_feat, [s.size(0) for s in feat] , batch_first=True)
    return packed_feat, pad_label

def collate_fn2(data):
    # Use zip() to unzip the data list into separate feature and label lists
    features, labels = zip(*data)

    # Sort the feature and label lists in descending order of length
    # Use the len() function as the key for sorting
    features = sorted(features, key=len, reverse=True)
    labels = sorted(labels, key=len, reverse=True)

    # Pad the feature and label sequences with zeros to make them the same length
    # Use the pad_sequence() function from the PyTorch library
    # Set batch_first=True to indicate that the batch dimension comes first in the tensor
    # Set padding_value=42 to pad with zeros (or any other desired value)
    padded_features = pad_sequence(features, batch_first=True)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=42)

    # Pack the padded feature sequences into a PackedSequence object
    # Use the pack_padded_sequence() function from the PyTorch library
    # Pass a list of the original lengths of the features to tell PyTorch how to unpack the sequence later
    packed_features = pack_padded_sequence(padded_features, [len(seq) for seq in features], batch_first=True)

    # Return the packed feature sequence and the padded label sequence
    return packed_features, padded_labels