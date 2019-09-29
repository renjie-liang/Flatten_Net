import re
from model.noiser.noise_layers.cropout import Cropout
from model.noiser.noise_layers.crop import Crop
from model.noiser.noise_layers.identity import Identity
from model.noiser.noise_layers.dropout import Dropout
from model.noiser.noise_layers.resize import Resize
from model.noiser.noise_layers.quantization import Quantization
from model.noiser.noise_layers.jpeg_compression import JpegCompression

from lib.config import cfg
import torch.nn as nn
import numpy as np

class BASE_Noiser(nn.Module):
    """
    This module allows to combine different noise layers into a sequential noise module. The
    configuration and the sequence of the noise layers is controlled by the noise_config parameter.
    """
    def __init__(self):
        super(BASE_Noiser, self).__init__()

        self.noise_layers = [Identity()]
        noise_layers = Noiser_list()
        device = cfg.TRAIN.DEVICE

        for layer in noise_layers:
            if type(layer) is str:
                if layer == 'JpegPlaceholder':
                    self.noise_layers.append(JpegCompression(device))
                elif layer == 'QuantizationPlaceholder':
                    self.noise_layers.append(Quantization(device))
                else:
                    raise ValueError(f'Wrong layer placeholder string in Noiser.__init__().'
                                     f' Expected "JpegPlaceholder" or "QuantizationPlaceholder" but got {layer} instead')
            else:
                self.noise_layers.append(layer)
        # self.noise_layers = nn.Sequential(*noise_layers)

    def forward(self, encoded_and_cover):
        # random_noise_layer = np.random.choice(self.noise_layers, 1)[0]
        random_noise_layer = self.noise_layers[cfg.Noiser.RANDOM_NUM ]
        # random_noise_layer = self.noise_layers[4]
        # print(random_noise_layer)
        # print(random_noise_layer)
        img_noise, img_orign = random_noise_layer(encoded_and_cover)
        return img_noise













def Noiser_list():
    layers = []
    noiser_list =  cfg.Noiser.SET_LIST
    split_commands =noiser_list.split('+')

    for command in split_commands:
        # remove all whitespace
        command = command.replace(' ', '')
        if command[:len('cropout')] == 'cropout':
            layers.append(parse_cropout(command))
        elif command[:len('crop')] == 'crop':
            layers.append(parse_crop(command))
        elif command[:len('dropout')] == 'dropout':
            layers.append(parse_dropout(command))
        elif command[:len('resize')] == 'resize':
            layers.append(parse_resize(command))
        elif command[:len('jpeg')] == 'jpeg':
            layers.append('JpegPlaceholder')
        elif command[:len('quant')] == 'quant':
            layers.append('QuantizationPlaceholder')
        elif command[:len('identity')] == 'identity':
            # We are adding one Identity() layer in Noiser anyway
            pass
        else:
            raise ValueError('Command not recognized: \n{}'.format(command))



    return layers



def parse_pair(match_groups):
    heights = match_groups[0].split(',')
    hmin = float(heights[0])
    hmax = float(heights[1])
    widths = match_groups[1].split(',')
    wmin = float(widths[0])
    wmax = float(widths[1])
    return (hmin, hmax), (wmin, wmax)


def parse_crop(crop_command):
    matches = re.match(r'crop\(\((\d+\.*\d*,\d+\.*\d*)\),\((\d+\.*\d*,\d+\.*\d*)\)\)', crop_command)
    (hmin, hmax), (wmin, wmax) = parse_pair(matches.groups())
    return Crop((hmin, hmax), (wmin, wmax))

def parse_cropout(cropout_command):
    matches = re.match(r'cropout\(\((\d+\.*\d*,\d+\.*\d*)\),\((\d+\.*\d*,\d+\.*\d*)\)\)', cropout_command)
    (hmin, hmax), (wmin, wmax) = parse_pair(matches.groups())
    return Cropout((hmin, hmax), (wmin, wmax))


def parse_dropout(dropout_command):
    matches = re.match(r'dropout\((\d+\.*\d*,\d+\.*\d*)\)', dropout_command)
    ratios = matches.groups()[0].split(',')
    keep_min = float(ratios[0])
    keep_max = float(ratios[1])
    return Dropout((keep_min, keep_max))

def parse_resize(resize_command):
    matches = re.match(r'resize\((\d+\.*\d*,\d+\.*\d*)\)', resize_command)
    ratios = matches.groups()[0].split(',')
    min_ratio = float(ratios[0])
    max_ratio = float(ratios[1])
    return Resize((min_ratio, max_ratio))
