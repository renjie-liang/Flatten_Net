import torch
import torch.nn as nn
import torch.optim as optim
from lib.config import cfg

def get_loss():

    img_loss_name = 'nn.'+ cfg.MODEL.IMG_LOSS_NAME
    wm_loss_name  = 'nn.'+ cfg.MODEL.WM_LOSS_NAME
    img_loss = eval(img_loss_name)()
    wm_loss = eval(wm_loss_name)()
    return  img_loss, wm_loss
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)