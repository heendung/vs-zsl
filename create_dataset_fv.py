from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch
import os

class Dataset:
    def __init__(self, split_name):
        
        if '1k' in split_name:
            self.images = torch.load(split_name + '_x.pt')
            self.labels = torch.load(split_name + '_y.pt')
        if 'our_2hop' in split_name:
            self.images = torch.load(os.path.join('/local/scratch/jihyung', split_name + '_x.pt'))
            self.labels = torch.load(os.path.join('/local/scratch/jihyung', split_name + '_y.pt'))
        elif 'our_3hop' in split_name:
            self.images = torch.load(split_name + 'x.pt')
            self.labels = torch.load(split_name + 'y.pt')
        elif 'our_all_data' in split_name:
            self.images = torch.load(split_name + 'x.pt')
            self.labels = torch.load(split_name + 'y.pt')

    def __len__(self):
    
        return len(self.images)

    def __getitem__(self, idx):
        
        img = self.images[idx]
        label = self.labels[idx]

        return img, label
