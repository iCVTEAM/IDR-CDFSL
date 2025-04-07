import torch
import torch.nn as nn
import torch.nn.functional as F

from model.registry import MODEL
from model.utils import min_max_scaler
from kmeans_pytorch import kmeans


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, block_size=1, is_maxpool=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride
        self.downsample = downsample
        self.block_size = block_size
        self.is_maxpool = is_maxpool
        self.maxpool = nn.MaxPool2d(stride)
        self.num_batches_tracked = 0

    def forward(self, x):
        self.num_batches_tracked += 1
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        if self.is_maxpool:
            out = self.maxpool(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(block, 64, stride=2)
        self.layer2 = self._make_layer(block, 128, stride=2)
        self.layer3 = self._make_layer(block, 256, stride=2)
        self.layer4 = self._make_layer(block, 512, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='leaky_relu')

    def _make_layer(self, block, planes, stride=1, block_size=1, is_maxpool=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layer = block(self.inplanes, planes, stride, downsample,
                      block_size, is_maxpool=is_maxpool)
        layers.append(layer)
        self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return [x2, x3, x4]

@MODEL.register
class Res10_Backbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        num_channel = 512
        self.n = config.n
        self.feature_extractor = ResNet(BasicBlock)
        
        # number of channels for the feature map, correspond to d in the paper
        self.d = num_channel
        self.way = 5
        self.scale = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)
        self.alpha = nn.Parameter(torch.FloatTensor([0.0]), requires_grad=True)
        
        self.cls = Classifier(config.num_classes)

    def get_feature_map(self, inp):
        batch_size = inp.size(0)
        feature_map = self.feature_extractor(inp)
        
        [x2, x3, x4] = feature_map
        x2 = x2.view(batch_size, self.d // 4, -1)
        x3 = x3.view(batch_size, self.d // 2, -1)
        x4 = x4.view(batch_size, self.d, -1)

        x2 = x2.permute(0, 2, 1).contiguous()
        x3 = x3.permute(0, 2, 1).contiguous()
        x4 = x4.permute(0, 2, 1).contiguous()
        
        return x4

    def forward_train(self, inp):
        feature_map = self.get_feature_map(inp)
        logits = self.cls.forward_train(feature_map)
        return logits * self.scale

    def forward(self, inp):
        feature_map = self.get_feature_map(inp)
        logits = self.cls(feature_map)
        return logits * self.scale
    
    def layer_score(self, feat, l):
        assert l in [0, 1, 2]

        if l == 0:
            return self.cls.org_score(feat, self.cls.low)
        elif l == 1:
            return self.cls.org_score(feat, self.cls.mid)
        elif l == 2:
            return self.cls.org_score(feat, self.cls.high)

    def get_mats_para(self):
        mats_para = []

        mats_para.append(self.cls.low.data)
        mats_para.append(self.cls.mid.data)
        mats_para.append(self.cls.high.data)

        return mats_para
    
    def set_mats_para(self, x, slc=False):
        ff4 = self.filtering_features(x, self.n, slc)
        self.cls.high.data = ff4

    def filtering_features(self, feat, n, slc):
        way = self.way
        d = feat.size(2)

        feat = feat.view(way, -1, d)
        if slc:
            n = min(n, feat.size(1))
            feat = torch.stack([kmeans(feat[i], n)[1] for i in range(way)])
        return feat

        

class Classifier(nn.Module):
    def __init__(self, num_classes, n=20):
        super().__init__()
        num_channel = 512

        self.n = n
        self.d = num_channel
        self.r = nn.Parameter(torch.zeros(2))
        self.num_classes = num_classes
        self.d_scale = [4, 2, 1]
        
        cat_mats = []
        for d_scale in self.d_scale:
            matrix = torch.randn(self.num_classes, self.n, self.d // d_scale)
            cat_mats.append(matrix)
        
        self.low = nn.Parameter(cat_mats[0], requires_grad=True)
        self.mid = nn.Parameter(cat_mats[1], requires_grad=True)
        self.high = nn.Parameter(cat_mats[2], requires_grad=True)

    def get_recon_dist(self, query, support):
        lam = support.size(1) / support.size(2)
        ct = support.permute(0, 2, 1)
        cct = support.matmul(ct)
        inv = (cct + torch.eye(cct.size(-1)).to(cct.device).unsqueeze(0).mul(lam)).inverse()
        ctinv = ct.matmul(inv).matmul(support)
        t_ctinv = query.matmul(ctinv)
        dist = (t_ctinv - query.unsqueeze(0)).pow(2).sum(2).permute(1, 0)
        return dist
    
    def org_score(self, feat, cat_mat):
        d = feat.size(-1)
        b, resolution = feat.size(0), feat.size(1)
        
        feat = feat.view(b * resolution, d)
        
        recon_dist = self.get_recon_dist(query=feat, support=cat_mat)
        recon_dist = recon_dist.view(b, resolution, cat_mat.size(0)).mean(1)

        logits = recon_dist.neg()

        return logits

    def forward_train(self, x):
        f2, f3, f4 = x

        logits2 = self.org_score(f2, self.low)
        logits3 = self.org_score(f3, self.mid)
        logits4 = self.org_score(f4, self.high)

        logits2 = min_max_scaler(logits2)
        logits3 = min_max_scaler(logits3)
        logits4 = min_max_scaler(logits4)
        
        return logits4

    def row_norm(self, matrix):
        row_min = matrix.min(1)[0].unsqueeze(1)
        norm_matrix = matrix - row_min
        row_sum = norm_matrix.sum(1).unsqueeze(1)
        norm_matrix = norm_matrix  / row_sum

        return norm_matrix

    def forward(self, x):
        logits = self.org_score(x, self.high)
        logits = min_max_scaler(logits)

        return logits
    