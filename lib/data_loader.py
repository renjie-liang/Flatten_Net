from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn

input_size = 5
output_size = 2

batch_size = 30
data_size = 100

from torch.utils.data import Dataset, DataLoader

class RandomDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len
def get_loader():
    return DataLoader(dataset=RandomDataset(input_size, data_size),
                         batch_size=batch_size, shuffle=True)

