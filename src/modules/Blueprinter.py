import torch

import tensortools, modules

class Blueprinter(torch.nn.Module, modules.NormalInit):

    def __init__(self, 
        C,
        W,
        H,
        i_embedsize,
        o_embedsize,
        scale_factor = None,
        scale_w = 1,
        scale_h = 1
    ):
        super(Blueprinter, self).__init__()
        
        if scale_factor is not None:
            scale_w = scale_h = scale_factor
        
        self.W = W
        self.H = H
        self.w = scale_w
        self.h = scale_h
        self.i_embed = torch.nn.Embedding(W*H, i_embedsize)
        self.o_embed = torch.nn.Parameter(
            torch.rand(1, o_embedsize, scale_w, scale_h).normal_(mean=0, std=1)
        )
        self.net = torch.nn.Conv2d(C + i_embedsize + o_embedsize, C, 1)

        self.init_weights(self.i_embed)
        self.init_weights(self.net)

    def get_init_targets(self):
        return [torch.nn.Embedding, torch.nn.Conv2d]

    def forward(self, X):
    
        '''
        
        Given:
            X - (N, C, W, H) Tensor
        
        Outputs:
            X' - (N, C, W', H') Tensor
                --> W' == W * self.w
                --> H' == H * self.h
        
        '''
        # Check pre-conditions
        N, C, W, H = X.size()
        assert W == self.W and H == self.H
        
        # Add global input positions
        i_pos = torch.arange(W*H).long().to(X.device)
        i_emb = self.i_embed(i_pos).transpose(0, 1) # D, W*H
        i_emb = i_emb.view(1, -1, W, H).repeat(N, 1, 1, 1)
        I_X = torch.cat([X, i_emb], dim=1) # N, C+D, W, H
        
        # Add local output positions
        o_pos = self.o_embed.repeat(N, 1, W, H)
        I_X = tensortools.stretch2d(I_X, self.w, self.h)
        O_X = torch.cat([I_X, o_pos], dim=1)
        
        return self.net(O_X)
        
