import torch.nn.functional as F
import torch

def cross_entropy(model, input, target):
    """
    Evaluate the cross entropy loss.
    :param model:
    :param input:
    :param target:
    loss function for the additional functionalities
    :return:
    """
    output = model(input)
    loss = F.cross_entropy(output, target)
    return loss, output, {}


def cross_entropy_func(model, input, target):
    return lambda: model(input), lambda pred: F.cross_entropy(pred, target)
