#!/usr/bin/python3

import torch
import util, modules, spiral

DEVICE = "cuda"
EPOCHS = 200
BATCHSIZE = 100

def to_tensor(arr):
    return torch.from_numpy(arr).float().to(DEVICE)

def flip(t, axis):
    c = t.size(axis)
    i = torch.arange(-1, -c-1, -1).long().to(t.device)
    return t.transpose(0, axis)[i].transpose(0, axis)

@util.main
def main():

    X_truth_arr, X_observe_arr, t_truth_arr, t_observe_arr = spiral.generate_spiral2d()

    X_truth, X_train, t_truth, t = tuple(
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

    X_truth = flip(X_truth, 1)
    X_train = flip(X_train, 1)

    dataloader = create_loader(BATCHSIZE, X_train)

    model = modules.VAE(
        input_size=2,
        hidden_size=20,
        z_size=10,
        kl_weight=0.1
    ).to(DEVICE)
    
    optim = torch.optim.Adam(model.parameters())
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim)

    for epoch in range(EPOCHS):

        model.train()

        n = e = 0.0

        for X, _ in dataloader:
            Xh = model(X, t)
            loss = model.loss(lossf, X, Xh)

            optim.zero_grad()
            loss.backward()
            optim.step()

            n += 1.0
            e += loss.item()

        with torch.no_grad():
            model.eval()

            Xh = model(X_train, t)
            loss = lossf(Xh, X)

            print("Epoch %d extrapolation loss: %.4f" % (epoch, loss.item()))

            sched.step(loss)
