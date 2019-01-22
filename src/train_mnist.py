#!/usr/bin/python3

import torch, itertools, tqdm

import model, modules, datasets, util

def mismatch(Yh, Y):
    return (Yh.max(dim=1)[1] == Y).float().mean().item()

@util.main
def main():

    DEVICE = ["cpu", "cuda"][torch.cuda.is_available()]
    BATCHSIZE = 128
    TESTBATCH = 512
    VALIDATION_SPLIT = 0.2
    PATIENCE = 30
    
    train_X, train_Y, test_X, test_Y, CLASSES, CHANNELS, IMAGESIZE = datasets.mnist.get(download=1)

    train_X, train_Y, valid_X, valid_Y = datasets.util.validation_split(VALIDATION_SPLIT, train_X, train_Y)

    train_loader = datasets.util.create_loader(BATCHSIZE, train_X, train_Y)
    valid_loader = datasets.util.create_loader(TESTBATCH, valid_X, valid_Y)
    test_loader  = datasets.util.create_loader(TESTBATCH, test_X,  test_Y)

    MODEL = torch.nn.Sequential(

        torch.nn.Conv2d(CHANNELS, 32, 5, padding=2),
        torch.nn.MaxPool2d(2), # 32 -> 16
        torch.nn.BatchNorm2d(32),
        torch.nn.ReLU(),

        torch.nn.Conv2d(32, 64, 3, padding=1),
        torch.nn.MaxPool2d(2), # 16 -> 8
        torch.nn.BatchNorm2d(64),
        torch.nn.ReLU(),

        model.OdeBlock(

            model.OdeConv2d(64, 64, 3, padding=1,
                pre=torch.nn.Sequential(
                    torch.nn.BatchNorm2d(64),
                    torch.nn.ReLU()
                )
            ),

            model.OdeConv2d(64, 64, 3, padding=1,
                pre = torch.nn.Sequential(
                    torch.nn.BatchNorm2d(64),
                    torch.nn.ReLU()
                ),
                post = torch.nn.Sequential(
                    torch.nn.BatchNorm2d(64)
                )
            )
            
        ),
        
        torch.nn.AvgPool2d(8), # 4 -> 1
        modules.Reshape(64),

        torch.nn.Linear(64, CLASSES),
    ).to(DEVICE)

    print("Parameters: %d" % sum(torch.numel(p) for p in MODEL.parameters() if p.requires_grad))

    lossf = torch.nn.CrossEntropyLoss()
    optim = torch.optim.SGD(MODEL.parameters(), lr=0.1, momentum=0.9)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim)

    best_validation_score = float("inf")
    inpatience_score = 0

    for epoch in itertools.count(1):

        MODEL.train()

        n = train_e = 0.0

        for X, Y in tqdm.tqdm(train_loader, ncols=80):
            X = X.to(DEVICE)
            Y = Y.to(DEVICE)

            Yh = MODEL(X)
            loss = lossf(Yh, Y)

            optim.zero_grad()
            loss.backward()
            optim.step()

            n += 1.0
            train_e += mismatch(Yh, Y)

        train_e /= n

        with torch.no_grad():

            MODEL.eval()

            n = valid_e = verr = 0.0

            for X, Y in valid_loader:
                X = X.to(DEVICE)
                Y = Y.to(DEVICE)

                Yh = MODEL(X)
                loss = lossf(Yh, Y)

                n += 1.0
                verr += loss.item()
                valid_e += mismatch(Yh, Y)

            sched.step(verr/n)
            valid_e /= n

            if valid_e < best_validation_score:
                inpatience_score = 0
                best_validation_score = valid_e
            else:
                inpatience_score += 1
                if inpatience_score > PATIENCE:
                    print("Validation score did not decrease for %d iterations. Terminating." % PATIENCE)
                    break

            n = test_e = 0.0

            for X, Y in test_loader:
                X = X.to(DEVICE)
                Y = Y.to(DEVICE)

                Yh = MODEL(X)

                n += 1.0
                test_e += mismatch(Yh, Y)

            test_e /= n

            print("Train: %.4f | Valid: %.4f | Test: %.4f" % (train_e, valid_e, test_e))
