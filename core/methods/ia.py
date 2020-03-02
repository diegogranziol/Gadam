# Set the model at the IA point. This is adapted from SWA of Izmailov et al, 2017.

import torch

from ..utils import set_weights


class IA(torch.nn.Module):
    def __init__(self, base=None, *args, **kwargs):
        super(IA, self).__init__()

        if base is not None:
            if isinstance(base, torch.nn.Module):
                # Check whether the base model is already initialised.
                import copy
                self.base_model = copy.deepcopy(base)
            else:
                self.base_model = base(*args, **kwargs)

            self.num_parameters = sum(param.numel() for param in self.base_model.parameters())
        else:
            self.base_model = None
            self.num_parameters = None

        # self.register_buffer('mean', torch.zeros(self.num_parameters))
        self.register_buffer('mean', None)
        self.register_buffer('n_models', torch.zeros(1, dtype=torch.long))

        # Initialize subspace
        self.model_device = 'cpu'

        # dont put subspace on cuda

    def cuda(self, device=None):
        self.model_device = 'cuda'
        self.base_model.cuda(device=device)

    def to(self, *args, **kwargs):
        self.base_model.to(*args, **kwargs)
        device, dtype, non_blocking = torch._C._nn._parse_to(*args, **kwargs)
        self.model_device = device.type

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    def collect_model(self, base_model, *args, **kwargs):
        # need to refit the space after collecting a new model
        w = torch.cat([param.detach().view(-1) for param in base_model.parameters()])
        if self.mean is None:
            self.mean = w.clone()
        else:
            self.mean.mul_(self.n_models.item() / (self.n_models.item() + 1.0))
            self.mean.add_(w / (self.n_models.item() + 1.0))
        self.n_models.add_(1)

    def set_ia(self):
        set_weights(self.base_model, self.mean, self.model_device)
