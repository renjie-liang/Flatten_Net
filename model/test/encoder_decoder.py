import torch
import torch.nn as nn
from lib.config import cfg

from model.encoder import *
from model.decoder import *
from model.noiser.noiser_moelding import BASE_Noiser


class EncoderDecoder(nn.Module):
    def __init__(self):
        super(EncoderDecoder, self).__init__()

        self.encoder = eval(cfg.MODEL.ENCODER_NAME)()
        self.noiser = eval(cfg.MODEL.NOSIER_NAME)()
        self.decoder = eval(cfg.MODEL.DECODER_NAME)()

    def forward(self, img, wm):
        img_wm = self.encoder(img, wm)
        img_noise = self.noiser([img_wm,img])
        wm_decode = self.decoder(img_noise)
        
        return img_wm,img_noise, wm_decode


    def to_stirng(self):
        return '{}\n{}'.format(str(self.encoder), str(self.decoder))
