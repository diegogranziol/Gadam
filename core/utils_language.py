# The corresponding utils functions for the NLP tasks
# N.B: since these are quite different from the vision tasks, this separate util file is created. However, every effort
# was made so that the interfacing does not change much even if the actual function implementation differs significant
# -ly.

import torch
import numpy as np
import h5py
import codecs
import six
import json
from core.methods.ia import IA


def train_epoch(model, loader, criterion, optimizer,
                epoch: int, seq_length: int,
                device=None,
                grad_clip: float = None,
                ):
    """
    Train one epoch on the character-level model task
    Args:
        model:
        loader:
        criterion:
        optimizer:
        epoch:
        seq_length:
        device:
        grad_clip:

    Returns:

    """
    traindata = loader.make_batches('train', 0 if epoch % 2 == 0 else seq_length // 2)
    totalloss = torch.tensor(0.)
    model.train()
    for iter_data in traindata.data:

        N = iter_data.inputs.size(0)
        T = iter_data.inputs.size(1)
        optimizer.zero_grad()
        if isinstance(model, IA):
            model.base_model.clear_states()
        else:
            model.clear_states()
        with torch.no_grad():
            preinputs = iter_data.preinputs.to(device).long()
            model(preinputs)
        inputs = iter_data.inputs.to(device).long()
        outputs = model(inputs)
        loss = criterion(outputs.view(N * T, -1), iter_data.outputs.to(device).long().view(N * T))
        loss.backward()
        if grad_clip is not None:
            for par in model.parameters():
                par.grad.clamp_(-grad_clip, grad_clip)
        optimizer.step()
        totalloss += loss.detach()
    return {
        'loss': totalloss.item() / traindata.batch_count
    }


def save_epoch(dir, epoch, model):
    model.save_model("%s_%d" % (dir, epoch))


def eval_epoch(model, loader, criterion, device=None):
    """
    Evaluate the (trained) model on the validation dataset
    Args:
        model:
        loader:
        criterion:
        device:

    Returns:

    """
    totalloss = torch.tensor(0.)
    # A quick and dirty fix for the SWA models
    if isinstance(model, IA):
        model.base_model.clear_states()
    else:
        model.clear_states()
    model.eval()
    valdata = loader.make_batches('val', shuffle=False)
    with torch.no_grad():
        for iter_data in valdata.data:
            N = iter_data.inputs.size(0)
            T = iter_data.inputs.size(1)
            if iter_data.preinputs is not None:
                model(iter_data.preinputs.to(device).long())
            outputs = model(iter_data.inputs.to(device).long())
            # print(outputs, iter_data.outputs)
            loss = criterion(outputs.view(N * T, -1), iter_data.outputs.to(device).long().view(N * T))
            totalloss += loss
    return {
        'loss': totalloss.item() / valdata.batch_count
    }


def preprocess(dataset: str,
               json_path: str = None,
               h5_path: str = None,
               val_frac: float = 0.1,
               test_frac: float = 0.1,
               verbose: bool = False,
               encoding='utf-8'):
    """
    This function is an adaptation of
    https://github.com/jcjohnson/torch-rnn/blob/master/scripts/preprocess.py
    It takes in txt raw file and pre-process (including conversion into json and h5 formats that the language model
    requires, and also partition the data into testing and validation sets, if applicable)
    Args:
        dataset: the path to the raw dataset
        json_path: the save path of json output. If none, will be saved to the same directory of the raw txt dataset,
        with only the extension name changed
        h5_path: the save path of h5 path. Rest ditto as above
        val_frac: fraction of the data to be partitioned as validation set
        test_frac: fraction of data to be partitioned as testing set
        verbose: whether to use verbose mode for debugging information output
        encoding: encoding of the raw txt file - default is 'utf-8'

    Returns:

    """
    token_to_idx = {}
    total_size = 0
    if json_path is None or h5_path is None:
        splitted_path = dataset.split(".")
        before_extension = ".".join(splitted_path[:-1])
        if json_path is None:
            json_path = before_extension + ".json"
        if h5_path is None:
            h5_path = before_extension + ".h5"

    with codecs.open(dataset, 'r', encoding) as f:
        for line in f:
            total_size += len(line)
            for char in line:
                if char not in token_to_idx:
                    token_to_idx[char] = len(token_to_idx) + 1

    # Now we can figure out the split sizes
    val_size = int(val_frac * total_size)
    test_size = int(test_frac * total_size)
    train_size = total_size - val_size - test_size

    if verbose:
        print('Total vocabulary size: %d' % len(token_to_idx))
        print('Total tokens in file: %d' % total_size)
        print('  Training size: %d' % train_size)
        print('  Val size: %d' % val_size)
        print('  Test size: %d' % test_size)

    # Choose the datatype based on the vocabulary size
    dtype = np.uint8
    if len(token_to_idx) > 255:
        dtype = np.uint32
    if verbose:
        print('Using dtype ', dtype)

    # Just load data into memory ... we'll have to do something more clever
    # for huge datasets but this should be fine for now
    train = np.zeros(train_size, dtype=dtype)
    val = np.zeros(val_size, dtype=dtype)
    test = np.zeros(test_size, dtype=dtype)
    splits = [train, val, test]

    # Go through the file again and write data to numpy arrays
    split_idx, cur_idx = 0, 0
    with codecs.open(dataset, 'r', encoding) as f:
        for line in f:
            for char in line:
                splits[split_idx][cur_idx] = token_to_idx[char]
                cur_idx += 1
                if cur_idx == splits[split_idx].size:
                    split_idx += 1
                    cur_idx = 0

    # Write data to HDF5 file
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('train', data=train)
        f.create_dataset('val', data=val)
        f.create_dataset('test', data=test)

    # For 'bytes' encoding, replace non-ascii characters so the json dump
    # doesn't crash
    if encoding is None:
        new_token_to_idx = {}
        for token, idx in six.iteritems(token_to_idx):
            if ord(token) > 127:
                new_token_to_idx['[%d]' % ord(token)] = idx
            else:
                new_token_to_idx[token] = idx
        token_to_idx = new_token_to_idx

    # Dump a JSON file for the vocab
    json_data = {
        'token_to_idx': token_to_idx,
        'idx_to_token': {v: k for k, v in six.iteritems(token_to_idx)},
    }
    with open(json_path, 'w') as f:
        json.dump(json_data, f)

