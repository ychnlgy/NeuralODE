#!/usr/bin/python3

import torch
import util, modules, spiral, tensortools

DEVICE = "cuda"
EPOCHS = 200

def to_tensor(arr):
    return torch.from_numpy(arr).float().to(DEVICE)

@util.main
def main():

    torch.manual_seed(1337)

    X_truth_arr, X_observe_arr, t_truth_arr, t_observe_arr = spiral.generate_spiral2d()

    X_truth, X, t_truth, t = tuple(
        map(
            to_tensor, 
            [
                X_truth_arr, 
                X_observe_arr, 
                t_truth_arr, 
                t_observe_arr
            ]
        )
    )

    X_truth = tensortools.flip(X_truth, 1)
    X = tensortools.flip(X, 1)

    model = modules.VAE(
        input_size=2,
        rnn_layers=1,
        rnn_hidden=25,
        hidden_size=20,
        z_size=4,
        kl_weight=0.1
    ).to(DEVICE)
    
    optim = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(EPOCHS):
        
        Xh = model(X, t)
        loss = model.loss()

        optim.zero_grad()
        loss.backward()
        optim.step()

        print("Epoch %d extrapolation loss: %.4f" % (epoch, loss.item()))
