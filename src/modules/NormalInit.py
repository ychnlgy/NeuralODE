import torch

class NormalInit:

    def init_weights(self, module, miu=0, std=0.02):
        targets = self.get_init_targets()
        t = type(module)
        if t in targets:
            module.weight.data.normal_(mean=miu, std=std)
        elif t in [torch.nn.Sequential, torch.nn.ModuleList]:
            for submod in module:
                self.init_weights(submod, miu, std)

    def get_init_targets(self):
        raise NotImplementedError
