import random
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from model.registry import FINETUNING
from kmeans_pytorch import kmeans
from torchvision import transforms
from extensions.PSG import PseudoSampleGenerator
from utils import accuracy
from torchvision.transforms import ToPILImage
from PIL import Image


import torch
def mmd_linear(f_of_X, f_of_Y):
    delta = f_of_X - f_of_Y
    loss = torch.mean(torch.mm(delta, torch.transpose(delta, 0, 1)))
    return loss


@FINETUNING.register
class AlignFinetuning():
    
    def __init__(self, config, model):
        self.model = model

        self.config = config
        self.way = config.dataset.way
        self.shot = config.dataset.shot
        self.query_shot = config.dataset.query_shot

        self.criterion = nn.CrossEntropyLoss()
        self.kl_div = nn.KLDivLoss(reduction='sum')

    def org_cat(self, support, n, feat=False):
        [f2, f3, f4] = self.model.get_feature_map(support)

        ff2 = self.filtering_features(f2, n)
        ff3 = self.filtering_features(f3, n)
        ff4 = self.filtering_features(f4, n)

        new_cat_mats = []
        new_cat_mats.append(nn.Parameter(ff2))
        new_cat_mats.append(nn.Parameter(ff3))
        new_cat_mats.append(nn.Parameter(ff4))

        if feat:
            return new_cat_mats, [f2, f3, f4]
        return new_cat_mats
    
    def get_recon_feat(self, base, feat):
        lam = base.size(0) / base.size(1)
        ct = base.permute(1, 0)
        cct = base.matmul(ct)
        inv = (cct+torch.eye(cct.size(-1)).to(cct.device).mul(lam)).inverse()
        ctinv = ct.matmul(inv).matmul(base)
        t_ctinv = feat.matmul(ctinv)
        return t_ctinv

    def layer_score(self, feat, cls, l):
        assert l in [0, 1, 2]

        if l == 0:
            return cls.org_score(feat, cls.low)
        elif l == 1:
            return cls.org_score(feat, cls.mid)
        elif l == 2:
            return cls.org_score(feat, cls.high)

    def align(self, x, model, gnn, s_proto):
        way = self.way
        shot = self.shot
        n_pseudo = 75

        x = x.cuda()
        xs = x[:way*shot].reshape(-1, *x.size()[1:])
        pseudo_q_genrator = PseudoSampleGenerator(way, shot, n_pseudo)
        loss_fun = nn.CrossEntropyLoss().cuda()
        for name, para in self.model.named_parameters():
            if 'bn' not in name:
                para.requires_grad = False
        opt = torch.optim.Adam([{'params': gnn.parameters()}])
        opt_bn = torch.optim.Adam([{'params': model.parameters()}])
        n_query = n_pseudo//way
        pseudo_set_y = torch.from_numpy(np.repeat(range(way), n_query)).cuda()

        pre = gnn.n_query
        gnn.n_query = n_query
        d = 512

        for ns in range(self.config.finetuning.steps):
            pseudo_set = pseudo_q_genrator.generate(xs).view(-1, *xs.shape[1:])

            model.train()

            for module in model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.momentum = 0.1 * 1 / (1 + torch.exp(torch.tensor(ns)))

            feat = model.get_feature_map(pseudo_set)
            feat = feat.view(way, -1, *feat.shape[1:])

            pseudo_support = feat[:, :shot].reshape(-1, *feat.size()[2:])
            pseudo_query = feat[:, shot:].reshape(-1, *feat.size()[2:])

            tf = pseudo_query.view(-1, d)
            sf_map = self.get_recon_feat(s_proto, tf).view(*pseudo_query.shape)
            
            logits = model.cls(pseudo_query)
            pred = torch.softmax(logits, dim=1).view(way, n_query, way)

            logits2 = model.cls(sf_map)
            pred2 = torch.softmax(logits2, dim=1).view(way, n_query, way)

            supp_last_vec = pseudo_support.mean(1).view(way, shot, d)
            qury_last_vec = pseudo_query.mean(1).view(way, n_query, d)
            set_feat = torch.cat((supp_last_vec, qury_last_vec), dim=1)
            score_gnn = gnn.set_forward(set_feat, pred)
            loss1 = loss_fun(score_gnn, pseudo_set_y.long())

            qury_last_vec = sf_map.mean(1).view(way, n_query, d)
            set_feat = torch.cat((supp_last_vec, qury_last_vec), dim=1)
            score_gnn2 = gnn.set_forward(set_feat, pred2)
            loss2 = loss_fun(score_gnn2, pseudo_set_y.long())
            loss = loss1 + loss2

            opt.zero_grad()
            loss.backward(retain_graph=True)
            opt.step()

            pred = pred.view(-1, way)
            pred2 = pred2.view(-1, way)
            loss_kl1 = F.kl_div(pred.log(), pred2, reduction='batchmean')

            opt_bn.zero_grad()
            loss_kl1.backward()
            opt_bn.step()
            
            del pseudo_set, score_gnn, loss
        
        torch.cuda.empty_cache()
        gnn.n_query = pre

        supp_feat = model.get_feature_map(x[:way*shot])
        qury_feat = model.get_feature_map(x[way*shot:])
        model.set_mats_para(supp_feat, slc=True)
        model.cuda()
        
        return model, gnn, supp_feat, qury_feat


    def filtering_features(self, feat, n, cls=True):
        way = self.way
        d = feat.size(-1)

        if cls:
            feat = feat.view(way, -1, d)
            feat = torch.stack([kmeans(feat[i], n)[1] for i in range(way)])
        else:
            feat = kmeans(feat, n)[1]

        return feat