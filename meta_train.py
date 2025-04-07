import os
import torch
import torch.nn as nn
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader

from config import setup_config
from dataset.dataset import BaseDataset
from dataset.sampler import MetaTaskSampler
from torchvision import transforms
from model.registry import MODEL
from utils import PerformanceMeter, TqdmHandler, AverageMeter, accuracy
from extensions.gnnnet import GnnNet


class Trainer(object):
    """Test a model from a config which could be a training config.
    """

    def __init__(self):
        self.config = setup_config()
        self.report_one_line = True
        self.logger = self.get_logger()

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
        self.dataloader = self.get_dataloader(self.config.dataset)
        self.logger.info(f'Building model {self.config.model.name} ...')
        self.model = self.get_model(self.config.model)
        self.load_model(self.config.model)
        self.model = self.to_device(self.model, parallel=True)
        self.logger.info(f'Building model {self.config.model.name} OK!')
        self.gnn = GnnNet(self.config)
        self.gnn = self.to_device(self.gnn, parallel=True)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.gnn.parameters())
        self.log_root = os.path.join(self.config.experiment.log_dir, self.config.experiment.name)

        # build meters
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
            meter: AverageMeter() for meter in ['acc', 'loss']
        }
    
    def reset_average_meters(self):
        for meter in self.average_meters:
            self.average_meters[meter].reset()

    def get_model(self, config):
        """Build model in config
        """
        name = config.name
        model = MODEL.get(name)(config)
        return model
    
    def load_model(self, config):
        """Load model in config
        """
        assert 'load' in config and config.load != '', 'There is no valid `load` in config[model.load]!'
        state_dict = torch.load(config.load, map_location='cpu')
        model = state_dict['model']

        self.model = self.get_model(self.config.model)
        self.model = self.to_device(self.model, parallel=True)
        self.model.load_state_dict(model, strict=True)

        scale = self.model.scale.item()
        self.model.scale.data /= scale

    def get_transformer(self, config):
        image_size = config.image_size
        resize_size = config.resize_size
        return transforms.Compose([
                transforms.RandomResizedCrop((image_size, image_size)),
                transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.8,hue=0.1),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])

    def get_collate_fn(self):
        return None

    def get_dataset(self, config):
        path = os.path.join(config.meta_dir, 'train.txt')
        return BaseDataset(config.root_dir, path, transform=self.transformer)

    def get_dataloader(self, config):
        way = config.way
        shot = config.shot
        query_shot = config.query_shot
        n_eposide = config.n_eposide
        return DataLoader(self.dataset, batch_sampler=MetaTaskSampler(self.dataset, way, shot, query_shot, trial=n_eposide), 
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
        
    def save_model(self, name=None):
        model_name = self.config.model.name
        if name is None:
            path = os.path.join(self.log_root, f'{model_name}_epoch_{self.epoch + 1}.pth')
        else:
            path = os.path.join(self.log_root, name)
        torch.save({'model': self.model.state_dict(), 'gnn': self.gnn.state_dict()}, path)
        self.logger.info(f'model saved to: {path}')

    def train(self):
        self.logger.info(f'Testing model from {self.config.model.load}')
        self.epoch = 0
        self.save_model()
        for epoch in range(400):
            self.epoch = epoch
            self.reset_average_meters()
            self.meta_train()
            self.performance_meters['acc'].update(self.average_meters['acc'].avg)
            self.report()
            if (self.epoch + 1) % 50 == 0:
                self.save_model()
        

    def meta_train(self):
        val_bar = tqdm(self.dataloader, ncols=100, total=self.config.dataset.n_eposide)
        for data in val_bar:
            self.task(data)
            val_bar.set_description(f'Testing')
            val_bar.set_postfix(acc=self.average_meters['acc'].avg, loss=self.average_meters['loss'].avg)
    

    def task(self, data):
        images, labels = self.to_device(data['img']), self.to_device(data['label'])

        way = self.config.dataset.way
        shot = self.config.dataset.shot
        query_shot = self.config.dataset.query_shot

        support = images[:way*shot]
        s_label = torch.LongTensor([i // shot for i in range(shot * way)]).cuda()

        query = images[way*shot:]
        q_label = torch.LongTensor([i // query_shot for i in range(query_shot * way)]).cuda()

        with torch.no_grad():
            supp_feat = self.model.get_feature_map(support)
        self.model.set_mats_para(supp_feat, slc=True)
        self.model = self.to_device(self.model)

        with torch.no_grad():
            query_feat = self.model.get_feature_map(query)
        logits = self.model.cls(query_feat)

        pred = torch.softmax(logits, dim=1).view(way, query_shot, way)
        d = supp_feat[-1].size(-1)
        supp_last_vec = supp_feat[-1].mean(1).view(way, shot, d)
        qury_last_vec = query_feat[-1].mean(1).view(way, query_shot, d)
        set_feat = torch.cat((supp_last_vec, qury_last_vec), dim=1)
        score_gnn = self.gnn.set_forward(set_feat, pred)
        acc = accuracy(score_gnn, q_label, 1)

        self.average_meters['acc'].update(acc, query.size(0))

        loss = self.criterion(score_gnn, q_label)
        self.average_meters['loss'].update(loss.item(), query.size(0))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def report(self):
        metric_str = '  '.join([f'{metric}: {self.performance_meters[metric].current_value:.2f}'
                                for metric in self.performance_meters])
        self.logger.info(metric_str)


if __name__ == '__main__':
    tester = Trainer()
    print('lam = {:.2f}'.format(tester.config.model.lam))
    print('loss = loss_t + 0.1 * loss_supcon')
    tester.train()
