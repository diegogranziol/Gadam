# Logistic regression toy example

import torch.nn as nn
import torchvision.transforms as transforms

__all__ = ['Logistic']

#for MNIST change 28x28
#for CIFAR100 32x32x3
class _LogisticRegression(nn.Module):
    def __init__(self, num_classes=10, input_dim=28*28):
        super(_LogisticRegression, self).__init__()
        self.input_dim = input_dim
        self.layer = nn.Linear(self.input_dim, num_classes, bias=True)

    def forward(self, x):
        return self.layer(x.view(-1, self.input_dim))


class Logistic:
    base = _LogisticRegression
    args = list()
    kwargs = dict()
    # Default transform
    transform_train = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

