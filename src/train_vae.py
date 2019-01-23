#!/usr/bin/python3

import torch
import util, modules, spiral

DEVICE = "cuda"
EPOCHS = 200

def to_tensor(arr):
    return torch.from_numpy(arr).float().to(DEVICE)

@util.main
def main():

    X_truth_arr, X_observe_arr, t_truth_arr, t_observe_arr = spiral.generate_spiral2d(200)

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

    model = modules.VAE(input_size=2, hidden_size=32, z_size=16, output_size=2).to(DEVICE)
    lossf = torch.nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.01)#, momentum=0.9)

    for epoch in range(EPOCHS):

        model.train()

        Xh = model(X, t)
        loss = model.loss(lossf, X, Xh)

        optim.zero_grad()
        loss.backward()
        optim.step()

        with torch.no_grad():
            model.eval()

            Xh = model(X, t_truth)
            loss = lossf(Xh, X_truth)

            print("Epoch %d extrapolation loss: %.4f" % (epoch, loss.item()))
