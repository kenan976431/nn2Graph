import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, 
                 input_size=100,
                 hidden_size=128,
                 num_layers=2,
                 num_classes=10,
                 bidirectional=False,
                 dropout=0.3):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.directions = 2 if bidirectional else 1
        
        # LSTM structure
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # classifier
        self.fc = nn.Linear(hidden_size * self.directions, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # init hidden state
        h0 = torch.zeros(self.num_layers * self.directions, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * self.directions, x.size(0), self.hidden_size).to(x.device)
        
        # forward propagation
        out, _ = self.lstm(x, (h0, c0))  # out: (batch_size, seq_length, hidden_size*directions)
        out = self.dropout(out[:, -1, :]) # get the last time step output
        out = self.fc(out)
        return out


def LSTM_small():
    return LSTM(hidden_size=64, num_layers=1, dropout=0.2)

def LSTM_large():
    return LSTM(hidden_size=256, num_layers=3, bidirectional=True, dropout=0.4)