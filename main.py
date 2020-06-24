#
# For licensing see accompanying LICENSE.txt file.
# Copyright (C) 2019-2020 Apple Inc. All Rights Reserved.
#
import os
import time
import torch
import pprint
import argparse
import contextlib
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from copy import deepcopy

from models.jit import IndependentJITApproximator, IndependentVectorizedApproximator
from models.dab import DAB, SignumWithMargin, View
from datasets.sort import SortLoader

parser = argparse.ArgumentParser(description='DAB Sort Dense Example')

# Task parameters
parser.add_argument('--batch-size', type=int, default=1024, metavar='N',
                    help='input batch size for training (default: 1024)')
parser.add_argument('--epochs', type=int, default=15000,
                    help='minimum number of epochs to train (default: 10000)')
parser.add_argument('--sequence-length', type=int, default=10,
                    help='size of each sequence to use for sorting (default: 10)')

# Model related
parser.add_argument('--latent-size', type=int, default=256,
                    help='latent layer size (default: 256)')
parser.add_argument('--dab-gamma', type=float, default=10.0,
                    help='the weighting for the DAB loss (default: 10)')
parser.add_argument('--approximator-type', type=str, default='batch',
                    help='batch [does all at once] or independent [elem-by-elem] approximator (default: batch)')

# Optimization related
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate (default: 1e-4)')
parser.add_argument('--optimizer', type=str, default="adam",
                    help="specify optimizer (default: adam)")

# Device /debug stuff
parser.add_argument('--debug-step', action='store_true', default=False,
                    help='only does one step of the execute_graph function per call instead of all minibatches')
parser.add_argument('--seed', type=int, default=None,
                    help='seed for numpy and pytorch (default: None)')
parser.add_argument('--ngpu', type=int, default=1,
                    help='number of gpus available (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--half', action='store_true', default=False,
                    help='enables half precision training')
parser.add_argument('--plot', action='store_true', default=False,
                    help='show plots when done')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    torch.backends.cudnn.benchmark = True

if args.plot:
    import matplotlib.pyplot as plt

# set a fixed seed for GPUs and CPU
if args.seed is not None:
    print("setting seed %d" % args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)


def all_or_none_accuracy(preds, targets, dim=-1):
    """ Gets the accuracy of the predicted sequence.

    :param preds: model predictions
    :param targets: the true targets
    :param dim: dimension to operate over
    :returns: scalar value for all-or-none accuracy
    :rtype: float32

    """
    preds_max = preds.data.max(dim=dim)[1]  # get the index of the max log-probability
    assert targets.shape == preds_max.shape, \
        "target[{}] shape does not match preds[{}]".format(targets.shape, preds_max.shape)
    targ = targets.data
    return torch.mean(preds_max.eq(targ).cpu().all(dim=dim).type(torch.float32))


def build_optimizer(model, args):
    """ helper to build the optimizer and wrap model

    :param model: the model to wrap
    :returns: optimizer wrapping model provided
    :rtype: nn.Optim

    """
    optim_map = {
        "rmsprop": optim.RMSprop,
        "adam": optim.Adam,
        "adadelta": optim.Adadelta,
        "sgd": optim.SGD,
        "lbfgs": optim.LBFGS
    }
    return optim_map[args.optimizer.lower().strip()](
        model.parameters(), lr=args.lr
    )


def build_model(args):
    """ Builds the approximator and the model with the DAB

    :param args: argparse
    :returns: model
    :rtype: nn.Sequential

    """
    if args.approximator_type == 'batch':
        approximator = nn.Sequential( # layers
            nn.Linear(args.latent_size, args.latent_size // 2),
            nn.Tanh(),
            nn.Linear(args.latent_size // 2, args.latent_size // 2),
            nn.Tanh(),
            nn.Linear(args.latent_size // 2, args.latent_size)
        )
    elif args.approximator_type == 'independent':
        approximator = IndependentVectorizedApproximator(args.latent_size,
                                                         activation=nn.Tanh)
    elif args.approximator_type == 'independent_jit':
        approximator = IndependentJITApproximator(args.latent_size,
                                                  args.sequence_length,
                                                  activation=nn.Tanh)
    else:
        raise Exception("unknown approximator type, specify independent or batch")

    model = nn.Sequential( # even more layers
        View([-1, args.sequence_length]),
        nn.Linear(args.sequence_length, args.latent_size // 2),
        nn.Tanh(),
        nn.Linear(args.latent_size // 2, args.latent_size // 2),
        nn.Tanh(),
        nn.Linear(args.latent_size // 2, args.latent_size),
        nn.Tanh(),
        DAB(
            approximator=approximator, 
            hard_layer=SignumWithMargin()
        ),
        nn.Linear(args.latent_size, args.sequence_length * args.sequence_length),
        View([-1, args.sequence_length, args.sequence_length])
    )
    print(model)

    return model.cuda() if args.cuda else model


def build_dataloader(args):
    """ Helper to build the data dataloader that houses
        both the train and test pytorch Dataloaders

    :param args: argparse
    :returns: SortLoader
    :rtype: object

    """
    return SortLoader(batch_size=args.batch_size,
                      upper_bound_unif=1,
                      sequence_length=args.sequence_length,
                      transform=None,
                      target_transform=None,
                      num_samples=2000000)


def get_dab_loss(model):
    """ Simple helper to iterate a model and return the DAB loss.

    :param model: the full nn.Sequential or nn.Modulelist
    :returns: dab loss
    :rtype: torch.Tensor

    """
    dab_loss, dab_count = None, 0
    for layer in model:
        if isinstance(layer, DAB):
            dab_count += 1
            if dab_loss is None:
               dab_loss = layer.loss_function()
            else:
                dab_loss += layer.loss_function()

    return dab_loss / dab_count


@contextlib.contextmanager
def dummy_context():
    """ Simple helper to create a fake context scope.

    :returns: None
    :rtype: Scope

    """
    yield None


def execute_graph(epoch, model, loader, optimizer=None, prefix='test'):
    """ execute the graph; when 'train' is in the name the model runs the optimizer

    :param epoch: the current epoch number
    :param model: the torch model
    :param loader: the train or **TEST** loader
    :param optimizer: the optimizer
    :param prefix: 'train', 'test' or 'valid'
    :returns: loss scalar
    :rtype: float32

    """
    start_time = time.time()
    model.eval() if prefix == 'test' else model.train()
    assert optimizer is not None if 'train' in prefix or 'valid' in prefix else optimizer is None
    accuracy, loss, dab_loss, num_samples = 0., 0., 0., 0.

    # iterate over train and valid data
    for minibatch, labels in loader:
        minibatch = minibatch.cuda() if args.cuda else minibatch
        labels = labels.cuda() if args.cuda else labels
        if args.half:
            minibatch = minibatch.half()

        if 'train' in prefix:
            optimizer.zero_grad()                                                  # zero gradients

        with torch.no_grad() if prefix == 'test' else dummy_context():
            pred_logits = model(minibatch)                                         # get model predictions

            # classification + DAB loss
            dab_loss_t = get_dab_loss(model)
            classification_loss_t = torch.sum(F.cross_entropy(input=pred_logits, target=labels, reduction='none'), -1)
            loss_t = torch.mean(classification_loss_t + args.dab_gamma * dab_loss_t)

            loss += loss_t.item()                                                  # add to aggregate loss
            dab_loss += torch.mean(dab_loss_t).item()
            # print(loss, dab_loss)
            accuracy += all_or_none_accuracy(preds=F.softmax(pred_logits, dim=1),  # get accuracy value
                                             targets=labels, dim=1)
            num_samples += minibatch.size(0)

        if 'train' in prefix:                                                      # compute bp and optimize
            loss_t.backward()
            optimizer.step()

        if args.debug_step: # for testing purposes
            break


    # debug prints for a ** SINGLE ** sample, loss above is calculated over entire minibatch
    print('preds[0]\t =\t ', F.softmax(pred_logits[0], dim=1).max(dim=1)[1])
    print('targets[0]\t =\t ', labels[0])
    print('inputs[0]\t = ', minibatch[0])

    # reduce by the number of minibatches completed
    num_minibatches_completed = num_samples / minibatch.size(0)
    loss /= num_minibatches_completed
    dab_loss /= num_minibatches_completed
    accuracy /= num_minibatches_completed

    # print out verbose loggin
    print('{}[Epoch {}][{} samples][{:.2f} sec]: Loss: {:.4f}\tDABLoss: {:.4f}\tAccuracy: {:.4f}'.format(
        prefix, epoch, num_samples, time.time() - start_time,
        loss, dab_loss, accuracy * 100.0))

    # return this for early stopping if used
    return {
        'prefix': prefix,
        'num_samples': num_samples,
        'epoch': epoch,
        'loss': loss,
        'dab_loss': dab_loss,
        'accuracy': accuracy,
        'elapsed': time.time() - start_time,
    }


def train(epoch, model, optimizer, train_loader, prefix='train'):
    """ Helper to run execute-graph for the train dataset

    :param epoch: the current epoch
    :param model: the model
    :param test_loader: the train data-loader
    :param prefix: the default prefix; useful if we have multiple training types
    :returns: mean loss value
    :rtype: float32

    """
    return execute_graph(epoch, model, train_loader, optimizer, prefix='train')


def test(epoch, model, test_loader, prefix='test'):
    """ Helper to run execute-graph for the test dataset

    :param epoch: the current epoch
    :param model: the model
    :param test_loader: the test data-loader
    :param prefix: the default prefix; useful if we have multiple test types
    :returns: mean loss value
    :rtype: float32

    """
    return execute_graph(epoch, model, test_loader, prefix='test')


def run(args):
    """ Main entry-point into the program

    :param args: argparse
    :returns: None
    :rtype: None

    """
    loader = build_dataloader(args)           # houses train and test loader
    model = build_model(args)                 # the model itself
    optimizer = build_optimizer(model, args)  # the optimizer for the model

    # main training loop
    test_results = []
    for epoch in range(1, args.epochs + 1):
        train(epoch, model, optimizer, loader.train_loader)
        test_results.append(test(epoch, model, loader.test_loader))

    if args.plot:
        x = [r['epoch'] for r in test_results]
        y = [r['accuracy'] for r in test_results]
        plt.plot(x, y)
        plt.show()


if __name__ == "__main__":
    print(pprint.PrettyPrinter(indent=4).pformat(vars(args)))
    run(args)
