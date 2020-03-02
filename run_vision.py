import argparse, os,  sys, time, tabulate
import torch
from core import data, models, losses, utils
import base_optimizers
from core.methods.ia import IA
import numpy as np
import copy

parser = argparse.ArgumentParser(description='Training on Vision Problems')
parser.add_argument('--dir', type=str, required=True, help='training directory (default: None)')
parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset name (default: CIFAR10)')
parser.add_argument('--data_path', type=str, default=None, required=True, metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--use_test', dest='use_test', action='store_true',
                    help='use test dataset instead of validation (default: False)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size (default: 128)')
parser.add_argument('--num_workers', type=int, default=4, metavar='N', help='number of workers (default: 4)')
parser.add_argument('--model', type=str, default=None, required=True, metavar='MODEL',
                    help='model name (default: None)')
parser.add_argument('--resume', type=str, default=None, metavar='CKPT',
                    help='checkpoint to resume  from (default: None)')
parser.add_argument('--reset_resume', action='store_true')
parser.add_argument('--epochs', type=int, default=300, metavar='N', help='number of epochs to train (default: 200)')
parser.add_argument('--save_freq', type=int, default=25, metavar='N', help='save frequency (default: 25)')
parser.add_argument('--eval_freq', type=int, default=1, metavar='N', help='evaluation frequency (default: 5)')
parser.add_argument('--lr_init', type=float, default=0.1, metavar='LR', help='initial learning rate (default: 0.1)')
parser.add_argument('--lr_r', type=float, default=0.01, help='learning rate ratio. Default to 0.01')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
parser.add_argument('--wd', type=float, default=5e-4, help='weight decay (default: 1e-4)')
parser.add_argument('--no_schedule', action='store_true', help='store schedule')
parser.add_argument('--step_schedule', action='store_true')
parser.add_argument('--when', nargs="+", type=int, default=[-1],
                    help='When (which epochs) to divide the learning rate by 10 - accepts multiple')
parser.add_argument("--linear_annealing", action='store_true')

parser.add_argument('--optim',default='SGD', help='Optimiser to use')
parser.add_argument('--ia_start', type=float, default=161, metavar='N', help='IA start epoch number (default: 161)')
parser.add_argument('--ia_lr', type=float, default=0.02, metavar='LR', help='IA LR (default: 0.02)')
parser.add_argument('--ia_c_epochs', default=1, metavar='N',
                    help='IA model collection frequency/cycle length in epochs (default: 1)')
parser.add_argument('--ia_save_stats', action='store_true', help='Save the IA statistics')
parser.add_argument('--ia_resume', type=str, default=None, metavar='CKPT',
                    help='checkpoint to restor IA from (default: None)')
parser.add_argument("--verbose", action='store_true', help='verbose training mode')
parser.add_argument("--save_freq_weight_norm", type=int, default=1, metavar='N', help='save frequency of weight norm')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument("--partial", type=float, default=0.125, help='Partially adaptive parameter for Padam/GadamX')

# Experimental feature: incorporate Lookahead
parser.add_argument("--lookahead", action='store_true', help='Whether to use Lookahead plug-in')
parser.add_argument("--k", type=int, default=5, help='Number of lookahead steps')
parser.add_argument("--alpha", type=float, help='Alpha parameter in lookahead')

args = parser.parse_args()

# convert types
try:  args.ia_c_epochs = int(args.ia_c_epochs)
except ValueError: args.ia_c_epochs = float(args.ia_c_epochs)
args.ia = True if args.optim in ['SWA', 'Gadam', 'GadamX'] else False

args.dir = args.dir + args.dataset + '/' + args.model + "/" + args.optim
thing = ""
if args.no_schedule:
    thing += "_flat"
if args.step_schedule:
    thing += "_step"
if args.ia:
    args.dir += "IA"
    thing = "_ialr="+str(args.ia_lr)+'_iastart='+str(args.ia_start)
if args.lookahead:
    args.dir += "_LH" + "_alpha" + str(args.alpha) + "_k" + str(args.k)
args.dir += '/seed='+str(args.seed)+'_lr=' + str(args.lr_init) + thing + '_mom=' + str(args.momentum) + '_wd=' + str(
    args.wd) +'_numepochs=' + str(args.epochs) + '/'


if args.no_schedule and args.step_schedule:
    raise ValueError("Both no_schedule and step_schedule are turned on. Quitting due to ambiguity.")

args.device = None
if torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

print('Preparing directory %s' % args.dir)
os.makedirs(args.dir, exist_ok=True)
with open(os.path.join(args.dir, 'command.sh'), 'w') as f:
    f.write(' '.join(sys.argv))
    f.write('\n')

torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

print('Using model %s' % args.model)
model_cfg = getattr(models, args.model)

loaders, num_classes = data.loaders(
    dataset=args.dataset,
    path=args.data_path,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    transform_train=model_cfg.transform_train,
    transform_test=model_cfg.transform_test,
    use_validation=not args.use_test,
)

print('Preparing model')
print(*model_cfg.args, dict(**model_cfg.kwargs))
model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
if torch.cuda.device_count() > 1:
    print("Multi-GPU: Using " + str(torch.cuda.device_count()) + " GPUs for training.")
    import torch.nn as nn
    model = nn.DataParallel(model)
model.to(args.device)

if args.optim in ['SWA', 'Gadam', 'GadamX']:
    ia_model = IA(model_cfg.base, *model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
    ia_model.to(args.device)
print(args.optim + ' training')

if args.no_schedule: print("Constant learning rate schedule")
elif args.step_schedule: print("Piecewise constant learning rate schedule")
else: print("Linearly decaying learning rate schedule")


def schedule(epoch, total_epoch):
    t = epoch / (args.ia_start if args.ia else total_epoch)
    lr_ratio = args.ia_lr / args.lr_init if args.ia else args.lr_r
    if t <= 0.5:
        factor = 1.0
    elif t <= 0.9:
        factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    else:
        factor = lr_ratio
    return args.lr_init * factor


def schedule_piecewise_const(epoch):
    """
    Use a piecewise constant learning rate
    d: The proportion of new learning rate to the initial learning rate. 0.5 means halving the learning rate
    r: frequency of reducing the learning rate. e.g. 40: reducing learning rate every 40 epochs
    """
    if args.ia:
        if epoch in args.when and epoch < args.ia_start:
            args.lr_init /= 10
        elif epoch >= args.ia_start:
            return args.ia_lr
    elif epoch in args.when:
        args.lr_init /= 10
    return args.lr_init


def schedule_variant(epoch):
    """A variant of the linear learning rate schedule"""

    def _linear_annealing(epoch):
        t = epoch / args.epochs
        lr_ratio = 0.01
        if t <= 0.5:
            factor = 1.0
        elif t <= 0.9:
            factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
        else:
            factor = lr_ratio
        return args.lr_init * factor

    lr_tmp = _linear_annealing(epoch)
    if args.ia:
        if (epoch > args.ia_start) or lr_tmp <= args.ia_lr:
            return _linear_annealing(args.ia_start)
    return lr_tmp


weight_decay = args.wd
criterion = losses.cross_entropy

# Setting the optimizer
if args.optim == 'SGD' or args.optim == 'SWA':
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr_init,
        momentum=args.momentum,
        weight_decay=weight_decay
    )
elif args.optim in ['Adam', 'AdamW', 'Gadam']:
    optimizer = base_optimizers.Adam(
        model.parameters(),
        lr=args.lr_init,
        weight_decay=weight_decay,
        decoupled_wd=not args.optim == 'Adam'
    )
elif args.optim in ['Padam', 'PadamW', 'GadamX']:
    optimizer = base_optimizers.Padam(
        model.parameters(),
        lr=args.lr_init,
        weight_decay=weight_decay,
        partial=args.partial,
        decoupled_wd=not args.optim == 'Padam'
    )
else:
    raise NotImplementedError('Requested optimizer ' + args.optim + ' is not implemented.')

lh_enabled = False
if args.lookahead:
    base_optimizer = copy.deepcopy(optimizer)
    optimizer = base_optimizers.Lookahead(
        base_optimizer,
        k=args.k,
        alpha=args.alpha
    )
    lh_enabled = True
else:
    base_optimizer = None

start_epoch = 0
if args.resume is not None:
    print('Resume training from %s' % args.resume)
    checkpoint = torch.load(args.resume)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

if args.ia and args.ia_resume is not None:
    checkpoint = torch.load(args.ia_resume)
    ia_model.load_state_dict(checkpoint['state_dict'])

columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'te_loss', 'te_acc', 'te_top5_acc', 'time', 'mem_usage']
if args.ia:
    columns = columns[:-2] + ['IA_tr_loss', 'IA_tr_acc', 'IA_te_loss', 'IA_te_acc', 'IA_te_top5_acc'] + columns[-2:]
    ia_res = {'loss': None, 'accuracy': None, 'top5_accuracy': None}

utils.save_checkpoint(
    args.dir,
    start_epoch,
    epoch=start_epoch,
    state_dict=model.state_dict(),
    optimizer=optimizer.state_dict()
)

for epoch in range(start_epoch, args.epochs):
    time_ep = time.time()

    if not args.no_schedule:
        e = epoch - start_epoch if args.reset_resume else epoch
        total_e = args.epochs - start_epoch if args.reset_resume else args.epochs
        if args.step_schedule:
            lr = schedule_piecewise_const(e)
        elif args.linear_annealing:
            lr = schedule_variant(e)
        else:
            lr = schedule(e, total_e)
        utils.adjust_learning_rate(optimizer, lr)
    else:
        lr = args.lr_init

    if args.ia and args.ia_c_epochs < 1 and epoch >= args.ia_start:
        # If mode collection is more frequent than once per epoch
        train_res = utils.train_epoch(loaders['train'], model, criterion, optimizer, verbose=args.verbose,
                                      ia_model=ia_model, ia_batch_c=int(len(loaders['train']) * args.ia_c_epochs))
    else:
        train_res = utils.train_epoch(loaders['train'], model, criterion, optimizer, verbose=args.verbose)

    # update batch norm parameters before testing
    utils.bn_update(loaders['train'], model)
    test_res = utils.eval(loaders['test'], model, criterion)

    if epoch == 0 or epoch % args.eval_freq == args.eval_freq - 1 or epoch == args.epochs - 1:
        test_res = utils.eval(loaders['test'], model, criterion)
    else:
        test_res = {'loss': None, 'accuracy': None, 'top5_accuracy': None}

    if args.ia and (epoch + 1) > args.ia_start:
        if lh_enabled:
            # Switch from Lookahead (exponentially weighted moving average) to iterate averaging (simple average)
            lh_enabled = False
            optimizer = base_optimizer
        # If the frequency of collecting ia models is less than once per epoch - otherwise the models have been
        # collected already in the train_epoch call.
        if args.ia_c_epochs >= 1 and (epoch + 1 - args.ia_start) % args.ia_c_epochs == 0:
            ia_model.collect_model(model)

        if epoch == 0 or epoch % args.eval_freq == args.eval_freq - 1 or epoch == args.epochs - 1:
            ia_model.set_ia()
            utils.bn_update(loaders['train'], ia_model)
            train_res_ia = utils.eval(loaders['train'], ia_model, criterion, optimizer)
            ia_res = utils.eval(loaders['test'], ia_model, criterion)

        else:
            ia_res = {'loss': None, 'accuracy': None, "top5_accuracy": None}
            train_res_ia = {'loss': None, 'accuracy': None}

    else:
        train_res_ia = {'loss': None, 'accuracy': None}

    if epoch == args.epochs - 1:
        utils.save_checkpoint(
            args.dir,
            epoch + 1,
            epoch=epoch + 1,
            state_dict=model.state_dict(),
            optimizer=optimizer.state_dict()
        )
        if args.ia:
            utils.save_checkpoint(
                args.dir,
                epoch + 1,
                name='ia',
                epoch=epoch + 1,
                state_dict=ia_model.state_dict(),
            )

    time_ep = time.time() - time_ep
    memory_usage = torch.cuda.memory_allocated() / (1024.0 ** 3)

    values = [epoch + 1, lr, train_res['loss'], train_res['accuracy'], test_res['loss'], test_res['accuracy'],
              test_res['top5_accuracy'], time_ep, memory_usage]

    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
    if epoch % 40 == 0:
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]
    print(table)

