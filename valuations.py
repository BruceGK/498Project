import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Module):
    def __init__(self):
        super(Linear, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(6000, 3)
        )

    def forward(self, x):
        y = self.layers(x)
        return y


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(6000, 8000),
            nn.ReLU(),
            nn.Linear(8000, 8000),
            nn.ReLU(),
            nn.Linear(8000, 8000),
            nn.ReLU(),
            nn.Linear(8000, 8000),
            nn.ReLU(),
            nn.Linear(8000, 3)
        )

    def forward(self, x):
        y = self.layers(x)
        return y


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=6000, hidden_size=100, num_layers=3)
        self.hidden_to_out = nn.Linear(100, 3)

    def forward(self, x):
        # convert input to (seq * batch * feature) tensor
        x = x.view(1, 1, -1)
        # (hidden, cell_state) omitted
        hidden, _ = self.lstm(x)
        out = self.hidden_to_out(hidden)
        return out

def read_orderbooks(file):
    df = pd.read_pickle(file, compression='bz2')
    # SWA
    def swa(row):
        best_sells = deque([(0,0)]*10,10)
        best_buys = deque([(0,0)]*10,10)
        for price, volume in row.items():
            if volume < 0:
                best_buys.append((price, volume))
            if volume > 0:
                best_sells.append((price,volume))
        total = (0,0)
        for price, volume in best_sells:
            total = (total[0]+abs(volume),total[1]+abs(volume*price))
        for price, volume in best_buys:
            total = (total[0]+abs(volume),total[1]+abs(volume*price))

        return total[1]/total[0]
    # TODO: True Class

    return df.apply(swa, axis=1)
    
def train(data):
    linear_model = Linear()
    mlp_model = MLP()
    lstm_model = LSTM()

    # train linear model
    train_losses = []
    optimizer = torch.optim.Adam(linear_model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    linear_model.train()
    for i, (x, y) in enumerate(data):
        # Initialize optimizer
        optimizer.zero_grad()

        outputs = linear_model(x)
        loss = loss_fn(outputs, y)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        if i % 100 == 0:
            print(f'{i * 128} / {len(data)}')
