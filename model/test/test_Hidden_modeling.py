import torch
import torch.nn as nn
from model.encoder import *
from model.decoder import *
# import decoder
from model.noiser.noiser_moelding import BASE_Noiser

# from model.noiser.noiser_moelding import BASE_Noiser




import torch
import torch.nn as nn
import numpy as np


from lib.config import cfg
from model.test.encoderdecodr import EncoderDecoder

class test_Hidden_modeling:
    def __init__(self, configuration: HiDDenConfiguration, device: torch.device, noiser: Noiser, tb_logger):
        """
        :param configuration: Configuration for the net, such as the size of the input image, number of channels in the intermediate layers, etc.
        :param device: torch.device object, CPU or GPU
        :param noiser: Object representing stacked noise layers.
        :param tb_logger: Optional TensorboardX logger object, if specified -- enables Tensorboard logging
        """
        super(test_Hidden_modeling, self).__init__()

        self.encoder_decoder = EncoderDecoder(configuration, noiser).to(device)

        self.optimizer_enc_dec = torch.optim.Adam(self.encoder_decoder.parameters())


        self.device = cfg.TRAIN.DEVICE 

        ### ???? loss to device ?
        self.mse_loss = nn.MSELoss().to(device)



    def train_on_batch(self, batch: list):
        """
        Trains the network on a single batch consisting of images and messages
        :param batch: batch of training data, in the form [images, messages]
        :return: dictionary of error metrics from Encoder, Decoder, and Discriminator on the current batch
        """
        images, messages = batch

        batch_size = images.shape[0]
        self.encoder_decoder.train()
        
        # with torch.enable_grad():


        # --------------Train the generator (encoder-decoder) ---------------------
        # target label for encoded images should be 'cover', because we want to fool the discriminator

        encoded_images, noised_images, decoded_messages = self.encoder_decoder(images, messages)
        self.optimizer_enc_dec.zero_grad()

        g_loss_enc = self.mse_loss(encoded_images, images)

        g_loss_dec = self.mse_loss(decoded_messages, messages)
        g_loss = 0.7 * g_loss_enc + 1 * g_loss_dec

        g_loss.backward()
        self.optimizer_enc_dec.step()

        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        bitwise_avg_err = np.sum(np.abs(decoded_rounded - messages.detach().cpu().numpy())) / (
                batch_size * messages.shape[1])

        losses = {
            'loss           ': g_loss.item(),
            'encoder_mse    ': g_loss_enc.item(),
            'dec_mse        ': g_loss_dec.item(),
            'bitwise-error  ': bitwise_avg_err,
        }
        return losses, (encoded_images, noised_images, decoded_messages)
# BASE_Decoder