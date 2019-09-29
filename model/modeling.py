import torch
import torch.nn as nn

from model.net import *
from lib.config import cfg

class get_Model(nn.Module):
    def __init__(self):
        super(get_Model, self).__init__()

        self.net = eval(cfg.MODEL.NET_NAME)()

    def forward(self, x):
        x = self.net(x)
        return x


    def to_stirng(self):
        return '{}'.format(str(self.net))

# BASE_Decoder

