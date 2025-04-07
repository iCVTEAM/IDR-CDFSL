import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import torch.nn.functional as F

import numpy as np

from config import setup_config
from dataset.dataset import BaseDataset
from dataset.sampler import MetaTaskSampler
from torchvision import transforms
from model.registry import MODEL, FINETUNING
from extensions.gnnnet import GnnNet
from utils import PerformanceMeter, TqdmHandler, AverageMeter, accuracy, label_propagation


class Tester(object):
    """Test a model from a config which could be a training config.
    """

    def __init__(self):
        self.config = setup_config()
        self.report_one_line = True
        self.logger = self.get_logger()

        # set device. `config.experiment.cuda` should be a list of gpu device ids, None or [] for cpu only.
        self.device = self.config.experiment.cuda if isinstance(self.config.experiment.cuda, list) else []
        if len(self.device) > 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in self.device])
            self.logger.info(f'Using GPU: {self.device}')
        else:
            self.logger.info(f'Using CPU!')

        # build dataloader and model
        self.transformer = self.get_transformer(self.config.dataset.transformer)
        self.collate_fn = self.get_collate_fn()
        
        self.dataset = self.get_dataset(self.config.dataset)
        self.logger.info(f'Building dataset OK!')
        self.logger.info(f'Building dataloader ...')
        self.dataloader = self.get_dataloader(self.config.dataset)
        self.logger.info(f'Building model {self.config.model.name} ...')
        self.model = self.get_model(self.config.model)
        self.model = self.to_device(self.model, parallel=True)
        self.gnn = GnnNet(self.config)
        self.gnn = self.to_device(self.gnn, parallel=True)
        self.logger.info(f'Building model {self.config.model.name} OK!')
        self.finetuning = self.get_finetuning(self.config)

        self.performance_meters = self.get_performance_meters()
        self.average_meters = self.get_average_meters()

    def get_logger(self):
        logger = logging.getLogger()
        logger.handlers = []
        logger.setLevel(logging.INFO)

        screen_handler = TqdmHandler()
        screen_handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s'))
        logger.addHandler(screen_handler)
        return logger

    def get_performance_meters(self):
        return {
            metric: PerformanceMeter() for metric in ['acc']
        }

    def get_average_meters(self):
        return {
            meter: AverageMeter() for meter in ['pre', 'acc']
        }

    def get_model(self, config):
        """Build model in config
        """
        name = config.name
        model = MODEL.get(name)(config)
        return model
    
    def load_model(self, config):
        """Load model in config
        """
        model_source = torch.load(config.load, map_location='cpu', weights_only=True)['source_cls'].cuda()
        d = 512

        assert 'load' in config and config.load != '', 'There is no valid `load` in config[model.load]!'
        state_dict = torch.load(config.load, map_location='cpu', weights_only=True)
        model = state_dict['model']
        gnn = state_dict['gnn']
        self.s_proto = model_source.view(-1, d)

        model.pop('cls.low')
        model.pop('cls.mid')
        model.pop('cls.high')
        self.model = self.get_model(self.config.model)
        self.model = self.to_device(self.model, parallel=True)
        self.model.load_state_dict(model, strict=False)

        self.gnn = GnnNet(self.config)
        self.gnn = self.to_device(self.gnn, parallel=True)
        self.gnn.load_state_dict(gnn, strict=True)

        self.finetuning = self.get_finetuning(self.config)

        scale = self.model.scale.item()
        self.model.scale.data /= scale
    
    def get_finetuning(self, config):
        """Get finetuning method
        """
        name = config.model.finetuning
        finetuning = FINETUNING.get(name)(config, self.model)
        return finetuning

    def get_transformer(self, config):
        return transforms.Compose([
            transforms.Resize(size=config.resize_size),
            transforms.CenterCrop(size=config.image_size),
            transforms.ToTensor(),
        ])

    def get_collate_fn(self):
        return None

    def get_dataset(self, config):
        path = os.path.join(config.meta_dir, 'test.txt')
        return BaseDataset(config.root_dir, path, transform=self.transformer)

    def get_dataloader(self, config):
        way = config.way
        shot = config.shot
        query_shot = config.query_shot
        trial = config.trail
        return DataLoader(self.dataset, batch_sampler=MetaTaskSampler(self.dataset, way, shot, query_shot, trial=trial), 
                          num_workers=config.num_workers, pin_memory=False)

    def to_device(self, m, parallel=False):
        if len(self.device) == 0:
            m = m.to('cpu')
        elif len(self.device) == 1 or not parallel:
            m = m.to(f'cuda:{0}')
        else:
            m = m.cuda(self.device[0])
            m = torch.nn.DataParallel(m, device_ids=self.device)
        return m

    def get_model_module(self, model=None):
        """get `model` in single-gpu mode or `model.module` in multi-gpu mode.
        """
        if model is None:
            model = self.model
        if isinstance(model, torch.nn.DataParallel):
            return model.module
        else:
            return model

    def test(self):
        self.logger.info(f'Testing model from {self.config.model.load}')
        self.validate()
        self.performance_meters['acc'].update(self.average_meters['acc'].avg)
        self.report()

    def validate(self):
        acc_list = []
        val_bar = tqdm(self.dataloader, ncols=100, total=self.config.dataset.trail)
        for data in val_bar:
            acc = self.task_validate(data)
            val_bar.set_description(f'Testing')
            acc_value = self.average_meters['acc'].avg
            pre_value = self.average_meters['pre'].avg
            val_bar.set_postfix(acc=f'{acc_value:.3f}', pre=f'{pre_value:.3f}')
            acc_list.append(acc)
        mean, interval = self.get_score(acc_list)
        print(mean, interval)
    

    def task_validate(self, data):
        self.load_model(self.config.model)
        self.model.eval()
        images, labels = self.to_device(data['img']), self.to_device(data['label'])

        way = self.config.dataset.way
        shot = self.config.dataset.shot
        query_shot = self.config.dataset.query_shot

        support = images[:way*shot]
        s_label = torch.LongTensor([i // shot for i in range(shot * way)]).cuda()
        query = images[way*shot:]
        q_label = torch.LongTensor([i // query_shot for i in range(query_shot * way)]).cuda()

        self.model, self.gnn, supp_feat, qury_feat = self.finetuning.align(images, self.model, self.gnn, self.s_proto)

        with torch.no_grad():
            logits = self.model.cls(qury_feat)
            pred = torch.softmax(logits, dim=1).view(way, query_shot, way)
            d = supp_feat.size(-1)
            supp_last_vec = supp_feat.mean(1).view(way, shot, d)
            qury_last_vec = qury_feat.mean(1).view(way, query_shot, d)
            set_feat = torch.cat((supp_last_vec, qury_last_vec), dim=1)
            score_gnn = self.gnn.set_forward(set_feat, pred)

            pre = accuracy(score_gnn, q_label, 1)
            self.average_meters['pre'].update(pre, query.size(0))

            x_lp = qury_feat.mean(1).cpu().numpy()
            y_lp = torch.softmax(score_gnn, dim=1).cpu().numpy()
            logits_lp = label_propagation(x_lp, y_lp, way)

            lp_acc = accuracy(logits_lp, q_label, 1)
            self.average_meters['acc'].update(lp_acc, query.size(0))

        return pre

    def report(self):
        metric_str = '  '.join([f'{metric}: {self.performance_meters[metric].current_value:.2f}'
                                for metric in self.performance_meters])
        self.logger.info(metric_str)

    def get_score(self, acc_list):
        mean = np.mean(acc_list)
        interval = 1.96 * np.sqrt(np.var(acc_list) / len(acc_list))

        return mean, interval


if __name__ == '__main__':
    tester = Tester()
    tester.test()
