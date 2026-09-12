import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import random



class ActionDataset(Dataset):
    def __init__(self, data_path, data_scale=1.0):
        with open(data_path, 'rb') as file:
            self.data = pickle.load(file)
        self.data_scale = data_scale
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        cur_data = self.data[idx]
        return torch.tensor(cur_data)*self.data_scale

