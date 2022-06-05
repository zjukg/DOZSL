import numpy as np, sys, os, random, pdb, json, uuid, time, argparse
from pprint import pprint
import logging, logging.config
from collections import defaultdict as ddict
from ordered_set import OrderedSet
import os.path as osp
# PyTorch related imports
import torch
from torch.nn import functional as F
from torch.nn.init import xavier_normal_
from torch.utils.data import DataLoader
from torch.nn import Parameter
import torch.nn as nn
import shutil
np.set_printoptions(precision=4)

def ensure_path(path):
    if osp.exists(path):
        if input('path [{}] exists, remove? ([y]/n)'.format(path)) != 'n':
            shutil.rmtree(path)
            os.mkdir(path)
    else:
        os.mkdir(path)

def get_combined_results(left_results, right_results):
    results = {}
    count = float(left_results['count'])

    results['left_mr'] = round(left_results['mr'] / count, 5)
    results['left_mrr'] = round(left_results['mrr'] / count, 5)
    results['right_mr'] = round(right_results['mr'] / count, 5)
    results['right_mrr'] = round(right_results['mrr'] / count, 5)
    results['mr'] = round((left_results['mr'] + right_results['mr']) / (2 * count), 5)
    results['mrr'] = round((left_results['mrr'] + right_results['mrr']) / (2 * count), 5)

    for k in range(10):
        results['left_hits@{}'.format(k + 1)] = round(left_results['hits@{}'.format(k + 1)] / count, 5)
        results['right_hits@{}'.format(k + 1)] = round(right_results['hits@{}'.format(k + 1)] / count, 5)
        results['hits@{}'.format(k + 1)] = round(
            (left_results['hits@{}'.format(k + 1)] + right_results['hits@{}'.format(k + 1)]) / (2 * count), 5)
    return results


def get_param(shape):
    param = Parameter(torch.Tensor(*shape));
    xavier_normal_(param.data)
    return param


def com_mult(a, b):
    r1, i1 = a[..., 0], a[..., 1]
    r2, i2 = b[..., 0], b[..., 1]
    return torch.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim=-1)


def conj(a):
    a[..., 1] = -a[..., 1]
    return a


def cconv(a, b):
    return torch.irfft(com_mult(torch.rfft(a, 1), torch.rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))


def ccorr(a, b):
    return torch.irfft(com_mult(conj(torch.rfft(a, 1)), torch.rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))
