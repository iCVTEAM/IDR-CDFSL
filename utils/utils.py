import logging
import time
import random
import numpy as np
import torch
from tqdm import tqdm
from yacs.config import CfgNode as CN
from torchvision.transforms.functional import resized_crop
from sklearn.neighbors import NearestNeighbors
from scipy.linalg import sqrtm


class PerformanceMeter(object):
    """Record the performance metric during training
    """

    def __init__(self, higher_is_better=True):
        self.best_function = max if higher_is_better else min
        self.current_value = None
        self.best_value = None
        self.best_epoch = None
        self.values = []

    def update(self, new_value):
        self.values.append(new_value)
        self.current_value = self.values[-1]
        self.best_value = self.best_function(self.values)
        self.best_epoch = self.values.index(self.best_value)

    @property
    def value(self):
        return self.values[-1]


class AverageMeter(object):
    """Keep track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class AccIntervalMeter(object):
    """Keep track of val acc and interval.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.values = []
        self.avg = 0
        self.interval = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.sum += val
        self.count += 1
        self.values.append(val)
        self.avg = self.sum / self.count
        self.interval = 1.96 * np.sqrt(np.var(self.values) / self.count)


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


class TqdmHandler(logging.StreamHandler):
    def __init__(self):
        super(TqdmHandler, self).__init__()

    def emit(self, msg):
        msg = self.format(msg)
        tqdm.write(msg)
        time.sleep(1)


class Timer(object):

    def __init__(self):
        self.start = time.time()
        self.last = time.time()

    def tick(self, from_start=False):
        this_time = time.time()
        if from_start:
            duration = this_time - self.start
        else:
            duration = this_time - self.last
        self.last = this_time
        return duration


def build_config_from_dict(_dict):
    cfg = CN()
    for key in _dict:
        cfg[key] = _dict[key]
    return cfg


def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def rotrate_concat(inputs):
    out = None
    for x in inputs:
        x_90 = x.transpose(2,3).flip(2)
        x_180 = x.flip(2).flip(3)
        x_270 = x.flip(2).transpose(2,3)
        if out is None:
            out = torch.cat((x, x_90, x_180, x_270),0)
        else:
            out = torch.cat((out, x, x_90, x_180, x_270),0)
    return out

def label_propagation(x_lp, y_lp, way):
    neigh = NearestNeighbors(n_neighbors=10)
    neigh.fit(x_lp)
    d_lp, idx_lp = neigh.kneighbors(x_lp)
    d_lp = np.power(d_lp, 2)
    sigma2_lp = np.mean(d_lp)

    n_lp = len(y_lp)
    del_n = int(n_lp * (1.0 - 0.2))
    for i in range(way):
        yi = y_lp[:, i]
        top_del_idx = np.argsort(yi)[0:del_n]
        y_lp[top_del_idx, i] = 0

    w_lp = np.zeros((n_lp, n_lp))
    for i in range(n_lp):
        for j in range(10):
            xj = idx_lp[i, j]
            w_lp[i, xj] = np.exp(-d_lp[i, j] / (2 * sigma2_lp))
            w_lp[xj, i] = np.exp(-d_lp[i, j] / (2 * sigma2_lp))
    q_lp = np.diag(np.sum(w_lp, axis=1))
    q2_lp = sqrtm(q_lp)
    q2_lp = np.linalg.inv(q2_lp)
    L_lp = np.matmul(np.matmul(q2_lp, w_lp), q2_lp)
    a_lp = np.eye(n_lp) - 0.5 * L_lp
    a_lp = np.linalg.inv(a_lp)
    ynew_lp = np.matmul(a_lp, y_lp)

    ynew_lp = torch.tensor(ynew_lp).cuda()
    ynew_lp = ynew_lp.to(torch.float32)

    return ynew_lp