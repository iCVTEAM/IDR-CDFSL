import os
import torch
import torchvision
import pandas as pd
from PIL import Image
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm import tqdm

identity = lambda x: x

def webfg_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        img = img.convert('RGB')
        return img


def default_loader(path):
    img = Image.open(path)
    img = img.convert("RGB")
    return img


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, root, meta_path, transform=None, return_id=False, loader=default_loader):
        self.root = root

        try:
            self.images = pd.read_csv(
                meta_path, sep='$', names=['label', 'path'])
        except:
            self.images = pd.read_csv(
                meta_path, sep=' ', names=['label', 'path'])
        
        self.images_data = self.images.values
        self.transform = transform
        self.return_id = return_id
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            dict { img, label }: dict of img and label.
        """

        label, image_path = self.images_data[index]

        img = self.loader(os.path.join(self.root, image_path))

        if self.transform is not None:
            img = self.transform(img)

        data = {
            'img': img,
            'label': label
        }

        if self.return_id:
            data['id'] = index

        return data

    def __len__(self):
        return len(self.images)
