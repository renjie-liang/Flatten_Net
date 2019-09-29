
from lib.config import cfg
import os
import torch
import logging
import torch.nn as nn

def save_ckpt(output_dir, epoch_step, model, optimizer):
    epoch,step = epoch_step
    """Save checkpoint"""

    if cfg.TRAIN.NO_SAVE:
        return
    ckpt_dir = os.path.join(output_dir, 'ckpt')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    save_name = os.path.join(ckpt_dir, 'model_{}_{}.pth'.format(epoch, step))
    if isinstance(model, nn.DataParallel):
        model = model.module
    # TODO: (maybe) Do not save redundant shared params
    # model_state_dict = model.state_dict()
    torch.save({
        'epoch': epoch,
        'step': step,
        'iters_per_epoch': cfg.TRAIN.STEP_PER_EPOCH,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()}, save_name)
    logging.info('save model: %s', save_name)


# def load_ckpt(model, ckpt):
#     """Load checkpoint"""
#     mapping, _ = model.detectron_weight_mapping
#     state_dict = {}
#     for name in ckpt:
#         if mapping[name]:
#             state_dict[name] = ckpt[name]
#     model.load_state_dict(state_dict, strict=False)