import os
import time
import lib
from lib.config import cfg

def create_folder_for_run(runs_folder):
    if not os.path.exists('./runs'):
        os.makedirs('./runs')
    if not os.path.exists(runs_folder):
        os.makedirs(runs_folder)
    os.makedirs(os.path.join(runs_folder, 'checkpoints'))

    
    cfg.DATA_SET.SAVE_IMGS_DIR = os.path.join(runs_folder, 'imgs')
    cfg.DATA_SET.SAVE_WMS_DIR = os.path.join(runs_folder, 'wms')
    os.makedirs(cfg.DATA_SET.SAVE_IMGS_DIR)
    os.makedirs(cfg.DATA_SET.SAVE_WMS_DIR)
    # os.makedirs(os.path.join(this_run_folder, 'images'))
    return runs_folder
