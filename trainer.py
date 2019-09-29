import csv
import os
import logging
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import numpy as np
from collections import defaultdict

from lib.config import cfg
from lib.utilis.average_meter import AverageMeter
from lib.ssim import SSIM 
from lib.utilis.net import save_ckpt

from tqdm import tqdm


def Trainer(loader, model):
    losses_record = defaultdict(AverageMeter)

    optimizer = torch.optim.Adam(model.parameters())
    

    # loss = loss.to(cfg.TRAIN.DEVICE)

    train_img_DataLoder, val_img_DataLoder = loader

    cfg.TRAIN.STEP_PER_EPOCH = int(len(train_img_DataLoder.dataset)/cfg.TRAIN.BATCH_SIZE)

    cfg.VAL.STEP_PER_EPOCH   = int(len(val_img_DataLoder.dataset)/cfg.TRAIN.BATCH_SIZE)

    for epoch in range(cfg.TRAIN.START_EPOCH , cfg.TRAIN.EPOCHS + 1):
        logging.info('EPOCH: {}|{}'.format(epoch, cfg.TRAIN.EPOCHS))

        correct_train = 0
        correct_val = 0

        model.train()
        pbar = tqdm(enumerate(train_img_DataLoder), total=len(train_img_DataLoder))

        # TRAIN
        for batch_idx, (img, lable) in pbar:

            # print(cfg.Noiser.RANDOM_NUM)  
            img = img.to(cfg.TRAIN.DEVICE)
            lable = lable.to(cfg.TRAIN.DEVICE)

            with torch.enable_grad():

                # backward
                y = model(img)
                optimizer.zero_grad()

                total_loss = get_loss(y, lable)

                total_loss.backward()
                optimizer.step()


            pred = y.data.max(1, keepdim=True)[1]
            correct_train += pred.eq(lable.data.view_as(pred)).sum().item()


            losses_dic = {
                'total_loss'        : float(total_loss)
            }

            # save loss
            for loss_name, loss_val in losses_dic.items():
                losses_record[loss_name].update(loss_val)


            if batch_idx % 100 == 0:
                done = batch_idx * len(img)
                percentage = 100. * batch_idx / len(train_img_DataLoder)
                pbar.set_description(
                    f'Train Epoch: {epoch}  [{done:5}/{len(train_img_DataLoder.dataset)} ({percentage:3.0f}%)]  Loss: {total_loss.item():.6f}')




        if epoch % 50 == 49:
            save_ckpt(cfg.TRAIN.RUNS_FOLDER, [epoch, cfg.TRAIN.STEP_PER_EPOCH], model, optimizer)


                #write losses after every epoch

        ### -----------vel------------###
        val_losses_record = defaultdict(AverageMeter)
        model.eval()

        pbar = tqdm(enumerate(val_img_DataLoder), total=len(val_img_DataLoder))
        for batch_idx, (img, lable) in pbar:

            # print(cfg.Noiser.RANDOM_NUM)  
            img = img.to(cfg.TRAIN.DEVICE)
            lable = lable.to(cfg.TRAIN.DEVICE)

            with torch.no_grad():

                # backward
                y = model(img)
                optimizer.zero_grad()

                total_loss = get_loss(y, lable)

            pred = y.data.max(1, keepdim=True)[1]
            correct_val += pred.eq(lable.data.view_as(pred)).sum().item()


            losses_dic = {
                'total_loss'        : float(total_loss)
            }

            # save loss

            for loss_name, loss_val in losses_dic.items():
                val_losses_record[loss_name].update(loss_val)


            if batch_idx % 100 == 0:
                done = batch_idx * len(img)
                percentage = 100. * batch_idx / len(val_img_DataLoder)
                pbar.set_description(f'Val Epoch:   {epoch}  [{done:5}/{len(val_img_DataLoder.dataset)} ({percentage:3.0f}%)]  Loss: {total_loss.item():.6f}')


        if epoch % 50 == 49:
            save_ckpt(cfg.TRAIN.RUNS_FOLDER, [epoch, cfg.TRAIN.STEP_PER_EPOCH], model, optimizer)





        accuracy_train = 100. * correct_train / len(train_img_DataLoder.dataset)
        losses_record['accuracy'].update(accuracy_train)

        accuracy_val = 100. * correct_val / len(val_img_DataLoder.dataset)
        val_losses_record['accuracy'].update(accuracy_val)

        print(f'Accuracy: Train:{correct_train}/{len(train_img_DataLoder.dataset)} ({accuracy_train:.2f}%) \
                Val:{correct_val}/{len(val_img_DataLoder.dataset)} ({accuracy_val:.2f}%) \n')
        if not cfg.TRAIN.NO_SAVE:
            write_losses(os.path.join(cfg.TRAIN.RUNS_FOLDER,  cfg.TRAIN.NAME  + '_train.csv'), losses_record, epoch, 0)
            write_losses(os.path.join(cfg.TRAIN.RUNS_FOLDER,  cfg.TRAIN.NAME  + '_val.csv'), val_losses_record, epoch, 0)

        # torch.cuda.empty_cache()





def log_print_helper(losses_accu, log_or_print_func):
    max_len = max([len(loss_name) for loss_name in losses_accu])
    for loss_name, loss_value in losses_accu.items():
        log_or_print_func(loss_name.ljust(max_len + 4) + '{:.4f}'.format(loss_value.avg))

def write_losses(file_name, losses_accu, epoch, duration):
    with open(file_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if epoch == 1:
            row_to_write = ['epoch'] + [loss_name.strip() for loss_name in losses_accu.keys()] + ['duration']
            writer.writerow(row_to_write)
        row_to_write = [epoch] + ['{:.4f}'.format(loss_avg.avg) for loss_avg in losses_accu.values()] + [
            '{:.0f}'.format(duration)]
        writer.writerow(row_to_write)






def save_imgs_wms(img, img_wm, img_noise, wm, wm_decoded,epoch):
    save_imgs(img, img_wm, img_noise, epoch)
    save_wms(wm, wm_decoded, epoch)



def save_imgs(img, img_wm, img_noise, epoch):
    img = img[: cfg.DATA_SET.SAVE_IMGS, :, :, :]
    img_wm = img_wm[: cfg.DATA_SET.SAVE_IMGS, :, :, :]
    img_noise = img_noise[: cfg.DATA_SET.SAVE_IMGS, :, :, :]

    img_r = img_wm.clone() - img.clone()
    img_r = img_r * 5
    # scale values to range [0, 1] from original range of [-1, 1]
    stacke_img = torch.cat([img, img_wm, img_noise, img_r], dim=0)
    stacke_img = torch.nn.functional.interpolate(stacke_img, size=cfg.DATA_SET.SAVE_IMGS_SIZE)


    filename = os.path.join(cfg.DATA_SET.SAVE_IMGS_DIR, 'epoch_{}.png'.format(epoch))

    torchvision.utils.save_image(stacke_img, filename, nrow=img.shape[0],
        padding=2, 
        normalize=True, 
        range=(-1,1), 
        scale_each=False, 
        pad_value=0)





    return loss
def get_loss(y, lable):
    loss_fun = eval('F.'+ cfg.MODEL.LOSS_NAME)

    loss = loss_fun(y,lable)
    return loss
