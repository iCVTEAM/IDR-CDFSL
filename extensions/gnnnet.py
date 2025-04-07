import torch
import torch.nn as nn
import numpy as np
from extensions.gnn import GNN_nl
import torch.nn.functional as F
from extensions.backbone import Linear_fw, BatchNorm1d_fw

class GnnNet(nn.Module):
  FWT=False
  def __init__(self, config):
    super(GnnNet, self).__init__()
    self.n_way = config.dataset.way
    self.n_support = config.dataset.shot
    self.n_query = config.dataset.query_shot
    self.feat_dim = config.model.num_channel
    self.loss_fn = nn.CrossEntropyLoss()
    self.fc = nn.Sequential(nn.Linear(self.feat_dim, 128), nn.BatchNorm1d(128, track_running_stats=False))
    self.gnn = GNN_nl(128 + self.n_way, 96, self.n_way)
    self.method = 'GnnNet'

  def cuda(self):
    self.feature.cuda()
    self.fc.cuda()
    self.gnn.cuda()
    return self

  def set_forward(self, x, pred):
    x = x.cuda()
    z = self.fc(x.view(-1, *x.size()[2:]))
    z = z.view(self.n_way, -1, z.size(1))
    z_stack = [torch.cat([z[:, :self.n_support], z[:, self.n_support + i:self.n_support + i + 1]], dim=1).view(1, -1, z.size(2)) for i in range(self.n_query)]
    assert(z_stack[0].size(1) == self.n_way*(self.n_support + 1))
    scores = self.forward_gnn(z_stack, pred)
    return scores

  def forward_gnn(self, zs, pred):
    nodes = []
    for i, z in enumerate(zs):
      support_label = torch.tensor(np.repeat(range(self.n_way), self.n_support), dtype=torch.int64).unsqueeze(1)
      support_label = torch.zeros(self.n_way*self.n_support, self.n_way).scatter(1, support_label, 1).view(self.n_way, self.n_support, self.n_way).cuda()
      support_label = torch.cat([support_label, pred[:, i].view(self.n_way, 1, self.n_way)], dim=1)
      support_label = support_label.view(1, -1, self.n_way)
      g = torch.cat([z, support_label], dim=2)
      nodes.append(g)
    nodes = torch.cat(nodes, dim=0)
    scores = self.gnn(nodes)
    scores = scores.view(self.n_query, self.n_way, self.n_support + 1, self.n_way)[:, :, -1].permute(1, 0, 2).contiguous().view(-1, self.n_way)
    return scores

  def set_forward_loss(self, x):
    y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
    y_query = y_query.cuda()
    scores = self.set_forward(x)
    loss = self.loss_fn(scores, y_query)
    return scores, loss

