import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn

import language_model.data

from language_model.utils import batchify, get_batch, repackage_hidden
from core.methods.ia import IA

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data_path', type=str, default='data/penn/',
                    help='location of the data corpus')
parser.add_argument("--dataset", type=str, default='PTB')
parser.add_argument("--dir", type=str, help='location of the save result', required=True)
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, QRNN, GRU)')
parser.add_argument('--emsize', type=int, default=400,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1150,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=3,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=30,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=200,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=70,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.4,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.25,
                    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.65,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0.1,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0.5,
                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--save', type=str, default=randomhash + '.pt',
                    help='path to save the final model')
parser.add_argument('--alpha', type=float, default=2,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument("--partial", type=float, default=0.2, help='partially adaptive parameter. '
                                                                   'Only applicable if using padam optimizer and its variants.')
parser.add_argument("--decoupled_wd", action='store_true')
parser.add_argument('--average_start', type=float, default=None, )
parser.add_argument("--ia_lr", type=float, default=None)
parser.add_argument("--ia_c_batches", type=int, default=1, help='frequency of IA model collection (by *minibatches*)')
parser.add_argument('--resume', type=str, default='',
                    help='path of model to resume')
parser.add_argument("--resume_epoch", type=int, default=None, help='the epoch number to resume from')
parser.add_argument('--optimizer', type=str, default='ASGD',
                    help='optimizer to use')
parser.add_argument('--when', nargs="+", type=int, default=[-1],
                    help='When (which epochs) to divide the learning rate by 10 - accepts multiple')
parser.add_argument("--lr_decay_factor", type=float, default=10., help='Amount of learning rate decay')

args = parser.parse_args()
args.tied = True

# Generate a specific directory to save the results
args.dir += '/' + args.dataset + '/' + args.model + '/' + args.optimizer
args.dir += '/lr=' + str(args.lr) + '_wd=' + str(args.wdecay) + '_seed' + str(args.seed) + '/'
import os

if not os.path.exists(args.dir): os.makedirs(args.dir)
print('Preparing directory ' + args.dir)

save = args.dir + str(randomhash) + '.pt'
args.ia = False
if args.optimizer in ['Gadam', 'GadamX']: args.ia = True

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)


###############################################################################
# Load data
###############################################################################

def model_save(fn):
    with open(fn, 'wb') as f:
        torch.save([model, criterion, optimizer], f)


def model_load(fn):
    global model, criterion, optimizer
    with open(fn, 'rb') as f:
        model, criterion, optimizer = torch.load(f)

import os
import hashlib

fn = args.dir + 'corpus.{}.data'.format(hashlib.md5(args.data_path.encode()).hexdigest())
if os.path.exists(fn):
    print('Loading cached dataset...')
    corpus = torch.load(fn)
else:
    print('Producing dataset...')
    corpus = language_model.data.Corpus(args.data_path)
    torch.save(corpus, fn)

eval_batch_size = 10
test_batch_size = 1
train_data = batchify(corpus.train, args.batch_size, args)
val_data = batchify(corpus.valid, eval_batch_size, args)
test_data = batchify(corpus.test, test_batch_size, args)

###############################################################################
# Build the model
###############################################################################

from language_model.splitcross import SplitCrossEntropyLoss
from language_model.model import RNNModel

criterion = None

ntokens = len(corpus.dictionary)
model = RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.dropouth,  #
                 args.dropouti, args.dropoute, args.wdrop, args.tied)

###
start_epoch = 1
if args.resume:
    print('Resuming model ...')
    if args.resume_epoch is not None: start_epoch += args.resume_epoch
    model_load(args.resume)
    optimizer.param_groups[0]['lr'] = args.lr
    model.dropouti, model.dropouth, model.dropout, args.dropoute = args.dropouti, args.dropouth, args.dropout, args.dropoute
    if args.wdrop:
        from language_model.weight_drop import WeightDrop

        for rnn in model.rnns:
            if type(rnn) == WeightDrop:
                rnn.dropout = args.wdrop
            elif rnn.zoneout > 0:
                rnn.zoneout = args.wdrop
###

if not criterion:
    splits = []
    if ntokens > 500000:
        # One Billion
        # This produces fairly even matrix mults for the buckets:
        # 0: 11723136, 1: 10854630, 2: 11270961, 3: 11219422
        splits = [4200, 35000, 180000]
    elif ntokens > 75000:
        # WikiText-103
        splits = [2800, 20000, 76000]
    print('Using', splits)
    criterion = SplitCrossEntropyLoss(args.emsize, splits=splits, verbose=False)
###
if args.cuda:
    model = model.cuda()
    criterion = criterion.cuda()
###
params = list(model.parameters()) + list(criterion.parameters())
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
print('Args:', args)
print('Model total parameters:', total_params)
if args.ia:
    ia_model = IA(RNNModel, args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.dropouth,  #
                 args.dropouti, args.dropoute, args.wdrop, args.tied)
    if args.cuda:
        ia_model.cuda()
else:
    ia_model = None


###############################################################################
# Training code
###############################################################################

def evaluate(data_source, batch_size=10, eval_ia=False):
    # Turn on evaluation mode which disables dropout.
    if eval_ia and args.ia:
        ia_model.eval()
        m = ia_model.base_model
    else:
        m = model
    model.eval()
    if args.model == 'QRNN': m.reset()
    total_ia_loss = 0.
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    hidden = m.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, args, evaluation=True)
        if eval_ia and args.ia:
            ia_model.set_IA()
            output, hidden = ia_model(data, hidden)
            total_ia_loss += len(data) * criterion(ia_model.base_model.decoder.weight,
                                                     ia_model.base_model.decoder.bias, output, targets).data
            output, _ = model(data, hidden)
            total_loss += len(data) * criterion(model.decoder.weight, model.decoder.bias, output, targets).data
        else:
            output, hidden = model(data, hidden)
            total_loss += len(data) * criterion(model.decoder.weight, model.decoder.bias, output, targets).data
        hidden = repackage_hidden(hidden)
    print('Loss:', total_loss.item() / len(data_source))
    if args.ia and eval_ia:
        print('ia Loss:', str(total_ia_loss.item() / len(data_source)))
        return total_ia_loss.item() / len(data_source)
    return total_loss.item() / len(data_source)


def train(ia=False):
    """By default, if IA_model is supplied, every model at the end of a batch will be averaged."""
    # Turn on training mode which enables dropout.
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    batch, i = 0, 0
    while i < train_data.size(0) - 1 - 1:
        bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
        # Prevent excessively small or negative sequence lengths
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        # There's a very small chance that it could select a very long sequence length resulting in OOM
        # seq_len = min(seq_len, args.bptt + 10)

        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
        model.train()
        data, targets = get_batch(train_data, i, args, seq_len=seq_len)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        optimizer.zero_grad()

        output, hidden, rnn_hs, dropped_rnn_hs = model(data, hidden, return_h=True)
        raw_loss = criterion(model.decoder.weight, model.decoder.bias, output, targets)

        loss = raw_loss
        # Activiation Regularization
        if args.alpha: loss = loss + sum(
            args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
        # Temporal Activation Regularization (slowness)
        if args.beta: loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        if args.clip: torch.nn.utils.clip_grad_norm_(params, args.clip)
        optimizer.step()

        total_loss += raw_loss.data
        optimizer.param_groups[0]['lr'] = lr2
        #  cur_loss = np.nan
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss.item() / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
                epoch, batch, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'],
                              elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss), cur_loss / math.log(2)))
            total_loss = 0
            start_time = time.time()
        ###

        if ia and args.ia and batch % args.ia_c_batches == 0:
            ia_model.collect_model(model)
            # print(ia_model.n_models)
        batch += 1
        i += seq_len

    return {'train_loss': cur_loss, 'train_perplexity': math.exp(cur_loss),
                       'train_bpc': cur_loss / math.log(2)}


# Loop over epochs.
lr = args.lr
best_val_loss = []
stored_loss = 100000000

# At any point you can hit Ctrl + C to break out of training early.
try:
    optimizer = None
    from base_optimizers.adam import Adam
    from base_optimizers import Padam

    # Ensure the optimizer is optimizing params, which includes both the model's weights as well as the criterion's
    # weight (i.e. Adaptive Softmax)
    if args.optimizer == 'ASGD':
        optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=args.wdecay)
    elif args.optimizer in ['Adam', 'Gadam']:
        optimizer = Adam(params, lr=args.lr, weight_decay=args.wdecay, decoupled_wd=args.decoupled_wd)
    elif args.optimizer in ['Padam', 'GadamX']:
        optimizer = Padam(params, lr=args.lr, weight_decay=args.wdecay, decoupled_wd=args.decoupled_wd,
                          partial=args.partial)
    else:
        raise ValueError("Unknown optimizer " + str(args.optimizer))
    args.ia = False
    if args.optimizer in ['ASGD', 'Gadam', 'GadamX']:
        args.ia = True

    start_average = False
    
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start_time = time.time()
        val_loss = None
        val_loss2 = None
        if args.ia or args.optimizer == 'ASGD':
            if args.average_start and epoch >= args.average_start:
                start_average = True
                print('Starting average')

        train_stat = None
        if start_average and args.ia:
            train_stat = train(True)
        else:
            train_stat = train()
        if isinstance(optimizer, torch.optim.ASGD):
            tmp = {}
            for (prm, st) in optimizer.state.items():
                tmp[prm] = prm.clone().detach()
                prm.data = st['ax'].clone().detach()
                
            val_loss2 = evaluate(val_data, eval_ia=start_average, )
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                  'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                epoch, (time.time() - epoch_start_time), val_loss2, math.exp(val_loss2), val_loss2 / math.log(2)))
            print('-' * 89)

            if val_loss2 < stored_loss:
                model_save(save)
                print('Saving Averaged!')
                stored_loss = val_loss2
            #
            for (prm, st) in optimizer.state.items():
                prm.data = tmp[prm].clone().detach()

        else:
            val_loss = evaluate(val_data, eval_batch_size, eval_ia=start_average)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                  'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss), val_loss / math.log(2)))
            print('-' * 89)

            if val_loss < stored_loss:
                model_save(save)
                print('Saving model (new best validation)')
                stored_loss = val_loss

            # Switching Strategy for averaging - ASGD by default averages every iteration
            if args.optimizer == 'ASGD' and args.average_start is None \
                        and 't0' not in optimizer.param_groups[0] \
                        and (len(best_val_loss) > args.nonmono and val_loss > min(best_val_loss[:-args.nonmono])):
                print('Switching to ASGD - Automatically triggered')
                optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0.,
                                                 weight_decay=args.wdecay)
                start_average = True
            elif start_average and not isinstance(optimizer, torch.optim.ASGD):
                    print('Switching to ASGD - Manulally triggered')
                    optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0.,
                                                 weight_decay=args.wdecay)
            elif args.ia:
                if args.average_start is None and len(best_val_loss) > args.nonmono and \
                        val_loss > min(best_val_loss[:-args.nonmono]):
                    print('Iterate Averaging Activated')
                    start_average = True
            if epoch in args.when and not start_average:
                print('Saving model before learning rate decreased')
                model_save('{}.e{}'.format(args.dir, epoch))
                print('Dividing learning rate by' + str(args.lr_decay_factor))
                optimizer.param_groups[0]['lr'] /= args.lr_decay_factor

            # Setting the learning rate to ia lr (if specified) - else use the previous lr
            if start_average is True and args.ia_lr is not None and args.ia:
                optimizer.param_groups[0]['lr'] = args.ia_lr

            best_val_loss.append(val_loss)


except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
model_load(save)

# Run on test data.
test_loss = evaluate(test_data, test_batch_size, eval_ia=args.ia)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
    test_loss, math.exp(test_loss), test_loss / math.log(2)))
print('=' * 89)
