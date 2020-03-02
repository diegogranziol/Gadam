import os
from collections import defaultdict
import itertools
import tqdm
import torch
import numpy as np
import torch.nn.functional as F


def save_checkpoint(dir, index, name='checkpoint', **kwargs):
    filepath = os.path.join(dir, '%s-%05d.pt' % (name, index))
    state = dict(**kwargs)
    torch.save(state, filepath)


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def adjust_learning_rate_and_momentum(optimizer, lr, momentum):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        param_group['momentum'] = momentum
    return lr, momentum


def train_epoch(loader, model, criterion, optimizer, cuda=True, verbose=False, subset=None,
                ia_model=None, ia_batch_c=64, ):
    """
    Train the model with one pass over the entire dataset (i.e. one epoch)
    :param loader:
    :param model:
    :param criterion:
    :param optimizer:
    :param cuda:
    :param verbose:
    :param subset:
    :param backpacked_model: toggle to true if the model has additional backpack functionality
    :param ia_model: whether swag model is used collect the models
    :param ia_batch_c: interval of swag model collection * on batch level *
    :return:
    """
    loss_sum = 0.0
    stats_sum = defaultdict(float)
    correct_1 = 0.0
    correct_5 = 0.0
    verb_stage = 0

    num_objects_current = 0
    num_batches = len(loader)

    model.train()

    if subset is not None:
        num_batches = int(num_batches * subset)
        loader = itertools.islice(loader, num_batches)

    if verbose:
        loader = tqdm.tqdm(loader, total=num_batches)

    for i, (input, target) in enumerate(loader):
        if cuda:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        loss, output, stats = criterion(model, input, target)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        loss_sum += loss.data.item() * input.size(0)
        for key, value in stats.items():
            stats_sum[key] += value * input.size(0)

        #pred = output.data.argmax(1, keepdim=True)
        #correct += pred.eq(target.data.view_as(pred)).sum().item()
        _, pred = output.topk(5, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_1 += correct[0].view(-1).float().sum(0)
        correct_5 += correct[:5].view(-1).float().sum(0)

        num_objects_current += input.size(0)

        if verbose and 10 * (i + 1) / num_batches >= verb_stage + 1:
            print('Stage %d/10. Loss: %12.4f. Acc: %6.2f. Top 5 Acc: %6.2f' % (
                verb_stage + 1, loss_sum / num_objects_current,
                correct_1 / num_objects_current * 100.0,
                correct_5 / num_objects_current * 100.0
            ))
            verb_stage += 1
        # print(loss_sum / num_objects_current)
        if ia_model is not None and i % ia_batch_c == 0:
            ia_model.collect_model(model)

    correct_5 = correct_5.cpu()
    correct_1 = correct_1.cpu()
    return {
        'loss': loss_sum / num_objects_current,
        'accuracy': correct_1 / num_objects_current * 100.0,
        'top5_accuracy': correct_5 / num_objects_current * 100.0,
        'stats': {key: value / num_objects_current for key, value in stats_sum.items()}
    }


def eval(loader, model, criterion, cuda=True, verbose=False):
    loss_sum = 0.0
    correct_1 = 0.0
    correct_5 = 0.0
    stats_sum = defaultdict(float)
    num_objects_total = len(loader.dataset)

    model.eval()

    with torch.no_grad():
        if verbose:
            loader = tqdm.tqdm(loader)
        for i, (input, target) in enumerate(loader):
            if cuda:
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            if criterion.__name__ != 'cross_entropy_func':
                loss, output, stats = criterion(model, input, target)
            else:
                model_fn, loss_fn = criterion(model, input, target)
                output = model_fn()
                loss = loss_fn(output)
                stats = {}
            loss_sum += loss.item() * input.size(0)
            for key, value in stats.items():
                stats_sum[key] += value

            #pred = output.data.argmax(1, keepdim=True)
            #correct += pred.eq(target.data.view_as(pred)).sum().item()

            _, pred = output.topk(5, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            correct_1 += correct[0].view(-1).float().sum(0) / num_objects_total * 100.0
            correct_5 += correct[:5].view(-1).float().sum(0) / num_objects_total * 100.0

    correct_1 = correct_1.cpu()
    correct_5 = correct_5.cpu()

    return {
        'loss': loss_sum / num_objects_total,
        'accuracy': correct_1,
        'top5_accuracy': correct_5,
        'stats': {key: value / num_objects_total for key, value in stats_sum.items()}
    }


def predict(loader, model, verbose=False):
    predictions = list()
    targets = list()

    model.eval()

    if verbose:
        loader = tqdm.tqdm(loader)

    offset = 0
    with torch.no_grad():
        for input, target in loader:
            input = input.cuda(non_blocking=True)
            output = model(input)

            predictions.append(F.softmax(output, dim=1).cpu().numpy())
            targets.append(target.numpy())
            offset += input.size(0)

    return {
        'predictions': np.vstack(predictions),
        'targets': np.concatenate(targets)
    }


def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def bn_update(loader, model, verbose=False, subset=None, **kwargs):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.

        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    num_batches = len(loader)

    with torch.no_grad():
        if subset is not None:
            num_batches = int(num_batches * subset)
            loader = itertools.islice(loader, num_batches)
        if verbose:
            loader = tqdm.tqdm(loader, total=num_batches)

        for input, _ in loader:
            input = input.cuda(non_blocking=True)
            input_var = torch.autograd.Variable(input)
            b = input_var.data.size(0)

            momentum = b / (n + b)
            for module in momenta.keys():
                module.momentum = momentum

            model(input_var, **kwargs)
            n += b

    model.apply(lambda module: _set_momenta(module, momenta))


def set_weights(model, vector, device=None):
    offset = 0
    for param in model.parameters():
        param.data.copy_(vector[offset:offset + param.numel()].view(param.size()).to(device))
        offset += param.numel()


def _bn_train_mode(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.train()


def hess_vec(vector, loader, model, criterion, cuda=True, bn_train_mode=True):
    param_list = list(model.parameters())
    vector_list = []

    offset = 0
    for param in param_list:
        vector_list.append(vector[offset:offset + param.numel()].detach().view_as(param).to(param.device))
        offset += param.numel()

    model.eval()
    if bn_train_mode:
        model.apply(_bn_train_mode)

    model.zero_grad()
    N = len(loader.dataset)
    for input, target in loader:
        if cuda:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
        loss, _, _ = criterion(model, input, target)
        loss *= input.size()[0] / N

        grad_list = torch.autograd.grad(loss, param_list, create_graph=True)
        dL_dvec = torch.zeros(1)
        if cuda:
            dL_dvec = dL_dvec.cuda()
        for v, g in zip(vector_list, grad_list):
            dL_dvec += torch.sum(v * g)
        dL_dvec.backward()
        #print(param_list[0].grad.size())
    model.eval()
    return torch.cat([param.grad.view(-1) for param in param_list]).view(-1)


def nonggn_vec(vector, loader, model, criterion, cuda=True, bn_train_mode=True):
    """Compute the matrix-vector product between the Hessian noise matrix"""
    full_hess_vec_prod = hess_vec(vector, loader, model, criterion, cuda=cuda, bn_train_mode=bn_train_mode)
    full_ggn_vec_prod = gn_vec(vector, loader, model, criterion, cuda=cuda, bn_train_mode=bn_train_mode)
    return full_hess_vec_prod - full_ggn_vec_prod


def grad(loader, model, criterion, cuda=True, bn_train_mode=False):
    model.eval()
    if bn_train_mode:
        raise NotImplementedError
        model.apply(_bn_train_mode)

    model.zero_grad()
    N = len(loader.dataset)
    for input, target in loader:
        if cuda:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
        loss, _, _ = criterion(model, input, target)
        loss *= input.size()[0] / N
        loss.backward()

    return torch.cat([param.grad.view(-1) for param in model.parameters()]).view(-1)


def gn_vec(vector, loader, model, criterion, cuda=True, bn_train_mode=True):
    param_list = list(model.parameters())
    vector_list = []
    num_parameters = sum(p.numel() for p in param_list)


    offset = 0
    for param in param_list:
        vector_list.append(vector[offset:offset + param.numel()].detach().view_as(param).to(param.device))
        offset += param.numel()

    model.eval()
    if bn_train_mode:
        model.apply(_bn_train_mode)

    model.zero_grad()
    N = len(loader.dataset)
    Gv = torch.zeros(num_parameters, dtype=torch.float32, device="cuda" if cuda else "cpu")

    for input, target in loader:
        if cuda:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
        loss, output, _ = criterion(model, input, target)
        loss *= input.size()[0] / N

        Jv = R_op(output, param_list, vector_list)
        grad = torch.autograd.grad(loss, output, create_graph=True)
        HJv = R_op(grad, output, Jv)
        JHJv = torch.autograd.grad(
            output, param_list, grad_outputs=HJv, retain_graph=True)
        Gv += torch.cat([j.detach().view(-1) for j in JHJv])
    # model.eval()
    return Gv
    # return torch.cat([param.grad.view(-1) for param in param_list]).view(-1)



def R_op(y, x, v):
    """
    Compute the Jacobian-vector product (dy_i/dx_j)v_j. R-operator using the two backward diff trick
    :return:
    """
    if isinstance(y, tuple):
        ws = [torch.zeros_like(y_i).requires_grad_(True) for y_i in y]
    else:
        ws = torch.zeros_like(y).requires_grad_(True)
    jacobian = torch.autograd.grad(y, x, grad_outputs=ws, create_graph=True)
    Jv = torch.autograd.grad(jacobian, ws, grad_outputs=v, retain_graph=True)
    return tuple([j.detach() for j in Jv])


def _gn_vec(model, loss, output, vec, ):
    """Compute the Gauss-newton vector product
    """
    views = []
    offset = 0
    param_list = list(model.parameters())
    for param in param_list:
        views.append(vec[offset:offset + param.numel()].detach().view_as(param).to(param.device))
        offset += param.numel()

    vec_ = list(views)

    Jv = R_op(output, param_list, vec_)

    gradient = torch.autograd.grad(loss, output, create_graph=True)
    HJv = R_op(gradient, output, Jv)
    JHJv = torch.autograd.grad(
        output, param_list, grad_outputs=HJv, retain_graph=True)
    Gv = torch.cat([j.detach().flatten() for j in JHJv])
    return Gv


def grad(loader, model, criterion, cuda=True, bn_train_mode=False):
    model.eval()
    if bn_train_mode:
        raise NotImplementedError
        model.apply(_bn_train_mode)

    model.zero_grad()
    N = len(loader.dataset)
    for input, target in loader:
        if cuda:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
        loss, _, _ = criterion(model, input, target)
        loss *= input.size()[0] / N
        loss.backward()

    return torch.cat([param.grad.view(-1) for param in model.parameters()]).view(-1)


def save_weight_norm(dir, index, name, model):
    """Save the L2 and L-inf norms of the weights of a model"""
    filepath = os.path.join(dir, '%s-%05d.pt' % (name, index))

    w = torch.cat([param.detach().cpu().view(-1) for param in model.parameters()])
    l2_norm = torch.norm(w).numpy()
    linf_norm = torch.norm(w, float('inf')).numpy()
    np.savez(
        filepath,
        l2_norms=l2_norm,
        linf_norms=linf_norm
    )
