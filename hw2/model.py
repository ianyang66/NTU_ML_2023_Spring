import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence

class GRUClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_layers, output_dim, dropout, fc_dropout):
        super(GRUClassifier, self).__init__()
        self.rnn = nn.GRU(
            input_size = input_dim,
            hidden_size = hidden_dim,
            num_layers = hidden_layers,
            batch_first = True,
            bidirectional = True,
            dropout=dropout #0.25
        )
        self.fc = nn.Sequential(
            nn.Dropout(fc_dropout),#0.4
            nn.Linear(hidden_dim * 2, output_dim)
        )
        
    def forward(self, x):
        output, _ = self.rnn(x)
        unpacked_out = output
        if type(output) == torch.nn.utils.rnn.PackedSequence:
            # unpacked
            unpacked_out, _ = pad_packed_sequence(output, batch_first=True)
        # print(unpacked_out.shape)
        x = self.fc(unpacked_out)
        return x
    
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_layers, output_dim, dropout, fc_dropout):
        super(LSTMClassifier, self).__init__()
        self.rnn = nn.LSTM(
            input_size = input_dim,
            hidden_size = hidden_dim,
            num_layers = hidden_layers,
            batch_first = True,
            bidirectional = True,
            dropout = dropout
        )
        self.fc = nn.Sequential(
            nn.Dropout(fc_dropout),#0.4
            nn.Linear(hidden_dim * 2, output_dim)
        )
        
    def forward(self, x):
        output, _ = self.rnn(x)
        unpacked_out = output
        if type(output) == torch.nn.utils.rnn.PackedSequence:
            # unpacked
            unpacked_out, _ = pad_packed_sequence(output, batch_first=True)
        # print(unpacked_out.shape)
        x = self.fc(unpacked_out)
        return x


class BasicBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BasicBlock, self).__init__()

        # TODO: apply batch normalization and dropout for strong baseline.
        # Reference: https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html (batch normalization)
        #       https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html (dropout)
        self.block = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Dropout(0.3),
            nn.Linear(input_dim,512),
            nn.ReLU6(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, output_dim),
            nn.ReLU6(),
        )

    def forward(self, x):
        x = self.block(x)
        return x 

class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim=41, hidden_layers=1, hidden_dim=1024):
        super(Classifier, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU6(),
            #BasicBlock(input_dim, hidden_dim),
            *[BasicBlock(hidden_dim, hidden_dim) for _ in range(hidden_layers)],
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        x = self.fc(x)
        return x