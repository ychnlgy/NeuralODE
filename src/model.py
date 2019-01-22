import torch
import torchdiffeq

TOL = 1e-3
EMPTY = torch.nn.Sequential()

class OdeConv2d(torch.nn.Conv2d):

    def __init__(self, in_channels, *args, pre=EMPTY, post=EMPTY, **kwargs):
        '''

        Input:
            (inherits original parameters as torch.nn.Conv2d)
            pre - torch.nn.Sequential operation before convolution is applied. Default: identity operation.
            post - torch.nn.Sequential operation after convolution is applied. Default: identity operation.

        '''
        super(OdeConv2d, self).__init__(in_channels+1, *args, **kwargs)
        self.pre = pre
        self.post = post

    def forward(self, t, X):
        '''

        Input:
            t - torch Tensor of size 0 (single float value). Time point.
            X - torch Tensor of size (N, C, W, H). Input.

        Output:
            X' - torch.Tensor of size (N, C', W', H'). Partial calculation
                of the gradient estimation at t with initial value X.
        
        '''
        N, C, W, H = X.size()
        t = torch.ones(N, 1, W, H).to(t.device) * t
        X = torch.cat([t, self.pre(X)], dim=1)
        return self.post(super(OdeConv2d, self).forward(X))

class OdeSequential(torch.nn.Module):

    def __init__(self, *args):
        super(OdeSequential, self).__init__()
        self.odeconvs = torch.nn.ModuleList(args)

    def forward(self, t, X):
        for odeconv in self.odeconvs:
            X = odeconv(t, X)
        return X

class OdeBlock(torch.nn.Module):

    def __init__(self, *odeconvs, eval_times=[0, 1]):
        '''

        Input:
            odeconvs - list of OdeConv2d. Sequential operations for estimating
                the gradient at any time point t with initial condition X0.
            eval_times - list of int. Time points at which the ODE is solved.

        '''
        super(OdeBlock, self).__init__()
        self.f = OdeSequential(*odeconvs)
        self.register_buffer("t", torch.FloatTensor(eval_times))

    def forward(self, X, i=-1):
        '''

        Input:
            X - torch Tensor of size (N, C, W, H).
            i - int index corresponding to time point of ODE estimation of the solution.

        Output:
            X_ti - torch Tensor of size (N, C', W', H'). Estimation of X at time point t[i].

        '''
        return torchdiffeq.odeint_adjoint(self.f, X, self.t, rtol=TOL, atol=TOL)[i]
