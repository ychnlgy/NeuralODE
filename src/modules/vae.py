import torch

import torchdiffeq

import tensortools

class Encoder(torch.nn.RNN):

    def __init__(self, output_size, input_size, hidden_size, *args, **kwargs):
        super(Encoder, self).__init__(input_size, hidden_size, *args, **kwargs)
        self.compressor = torch.nn.Linear(hidden_size, output_size*2)
        self.output_size = output_size
    
    def forward(self, X):
        
        '''

        Description:
            Returns z0 sampled from a distribution with mean(z0) = 0 and std(z0) = 1,
            where z0 is the latent encoding of reverse sequences in X.

        Input:
            X - torch Tensor of size (N, S, D), where N is the batch size,
                S is the sequence length and D is the number of features.

        Output:
            z0 - torch Tensor of size (N, H//2), where H is the hidden
                layer size of this encoder.
            
        '''
        
        N, S, D = X.size()
        out, _ = super(Encoder, self).forward(tensortools.flip(X, 1))
        z0 = self.compressor(out[:,-1])

        # Save these values for computing the loss later.
        self._q_miu = z0[:,:self.output_size]
        self._q_std = z0[:,self.output_size:]
        assert self._q_std.size() == self._q_miu.size()
        
        eps = torch.randn(self._q_std.size()).to(z0.device)
        return eps * torch.exp(self._q_std/2.0) + self._q_miu

    def loss(self):
        
        '''

        Description:
            Returns a KL divergence of examples from distribution q
            from that of distribution p, which is a normal distribution
            centered at 0 with 1 standard deviation.

        Output:
            kl_score - torch Tensor of size 0, scaler loss.
        
        '''
        kl_score = -self._q_std + torch.exp(self._q_std) + self._q_miu**2
        return kl_score.mean()
    
class OdeFunction(torch.nn.Sequential):

    def forward(self, t, X):
        return super(OdeFunction, self).forward(X)

class Decoder(torch.nn.Module):

    def __init__(self, ode_function, deciphernet):
        
        '''

        Input:
            ode_function - OdeFunction, mapping D -> D.
            deciphernet - vanilla torch.nn.Sequential, mapping D -> C (output size).
            
        '''
        
        super(Decoder, self).__init__()
        self.ode_function = ode_function
        self.deciphernet = deciphernet

    def forward(self, z0, t):
        
        '''

        Description:
            Uses the trajectory of z0 to estimate values at each time point in t.

        Input:
            z0 - torch Tensor of shape (N, D). Initial latent value, obtained from encoder.
            t - torch Tensor of shape (S). Time points to calculate the trajectory of z.
        
        '''
        
        pred_z = torchdiffeq.odeint(self.ode_function, z0, t) # S, B, D
        pred_z = pred_z.transpose(0, 1) # B, S, D
        return self.deciphernet(pred_z)

class VAE(torch.nn.Module):
    
    def __init__(self, input_size, rnn_layers, rnn_hidden, hidden_size, z_size, kl_weight):
        super(VAE, self).__init__()
        self.encoder = Encoder(
            output_size = z_size,
            input_size = input_size,
            hidden_size = rnn_hidden,
            num_layers = rnn_layers,
            batch_first = True
        )

        self.decoder = Decoder(
            ode_function = OdeFunction(
                torch.nn.Linear(z_size, hidden_size),
                torch.nn.ELU(inplace=True),
                torch.nn.Linear(hidden_size, hidden_size),
                torch.nn.ELU(inplace=True),
                torch.nn.Linear(hidden_size, z_size)
            ),
            deciphernet = torch.nn.Sequential(
                torch.nn.Linear(z_size, hidden_size),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(hidden_size, input_size)
            )
        )

        self.lossf = torch.nn.MSELoss()
        self.kl_weight = kl_weight
    
    def forward(self, X, t):
        self._X = X
        z0 = self.encoder(X)
        self._Xh = self.decoder(z0, t)
        return self._Xh
    
    def loss(self):
        return self.lossf(self._Xh, self._X) + self.kl_weight * self.encoder.loss()
