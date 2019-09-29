import os
import sys
import logging
import yaml
import pprint
import pickle
import shutil
from tqdm import tqdm

import torch
import torch.nn as nn

from option import args
from trainer import Trainer
from collections import defaultdict

from lib.utilis.timer import Timer
from lib.utilis.misc_utilis import create_folder_for_run
from lib.config import cfg, cfg_from_file, cfg_from_list, cfg_from_args, assert_and_infer_cfg

from model.data_loader import * 
from model.modeling import get_Model

# from model.test.test_Trainer import test_Trainer
# from model.test.test_Hidden_Trainer import test_Hidden_Trainer
# test lib
# from model.test.test_modeling import get_test_Model
# from model.test.test_Hidden_modeling import test_Hidden_modeling

# from trainer import Trainer

# torch.manual_seed(args.seed)
# checkpoint = utility.checkpoint(args)

# if checkpoint.ok:
#     loader = data.Data(args)
#     model = model.Model(args, checkpoint)
#     loss = loss.Loss(args, checkpoint) if not args.test_only else None
#     t = Trainer(args, loader, model, loss, checkpoint)
#     while not t.terminate():
#         t.train()
#         t.test()

#     checkpoint.done()
def main():
    ### set divice
    if not args.cuda:
        cfg.TRAIN.DEVICE = torch.device('cpu')
    else :
        assert torch.cuda.is_available(), "Not enough GPU"
        #assert d < torch.cuda.device_count(), "Not enough GPU"
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(ids) for ids in args.device_ids])
        torch.backends.cudnn.benchmark=True
        cfg.CUDA = True
        cfg.TRAIN.DEVICE = torch.device('cuda:0')
        print("Let's use", torch.cuda.device_count(), "GPUs!")

    ### set config 
    cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg_from_args(args)
    # print_cfg(cfg)


    # assert_and_infer_cfg()



    if not cfg.TRAIN.NO_SAVE:

        run_folder = create_folder_for_run(cfg.TRAIN.RUNS_FOLDER)
        logging.basicConfig(level=logging.INFO,
                            format='%(message)s',
                            handlers=[
                                logging.FileHandler(os.path.join(run_folder, f'{cfg.TRAIN.NAME}.log')),
                                logging.StreamHandler(sys.stdout)
                            ])

        with open(os.path.join(run_folder, 'config_and_args.pkl'), 'wb') as f:
            blob = {'cfg': yaml.dump(cfg), 'args': args}
            pickle.dump(blob, f, pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(run_folder, 'args.txt'), 'w') as f:
            for item in vars(args):
                f.write(item+":"+str(getattr(args,item))+'\n')
        logging.info('×' * 40)

        shutil.copy(args.cfg_file, os.path.join(run_folder, cfg.TRAIN.NAME) + '_cfg')
        logging.info('save config and args in runs folder:\n %s' % run_folder)
        # if args.use_tfboard:
        #     tblogger = SummaryWriter(run_folder)

    else:
        logging.basicConfig(level=logging.INFO)
        # logger = logging.getLogger(__name__)
    # print('args:')
    # logging.info(pprint.pformat(vars(args)))
    # print('cfg:')
    # logging.info(yaml.dump(cfg.TRAIN))


    loader = get_CIFARLoader()

    model = get_Model()
    
    # model = nn.DataParallel(model)
    model.to(cfg.TRAIN.DEVICE)

    logging.info(model)
    
    Trainer(loader, model)
    # # from noise_argparser import NoiseArgParser


    # logging.info('HiDDeN model: {}\n'.format(model.module.to_stirng()))




    # timers = defaultdict(Timer)
    # # timers['roidb'].tic()
    # timers['roidb'].tic()
    # timers['roidb'].toc()
    # print( timers['roidb'].tic())
    # print('Takes {:.2f} sec(s) to construct roidb'.format(timers['roidb'].average_time))

    # if args.debug:
    #     cfg.SEM.INPUT_SIZE = [256, 256]
    #     args.no_save = True
    #     args.batch_size = 2
    #     train_size = 8
    #     args.num_epochs = 30
    #     cfg.TRAIN.DATASETS = ('steel_train_on_val')





def print_cfg_aux(c):
    c = str(c.immutable)
    c = c.split('{')
    c = c[1].split('}')
    c = c[0]
    c = c.replace(", ",'\n')
    print(c)

def print_cfg(cfg):
    print()
    print('*' * 5, 'MODEL', '*' * 5)
    print_cfg_aux(cfg.MODEL)
    print()
    print('*' * 5, 'DATA_SET', '*' * 5)
    print_cfg_aux(cfg.DATA_SET)
    print()
    print('*' * 5, 'TRAIN', '*' * 5)
    print_cfg_aux(cfg.TRAIN)

    







if __name__ == '__main__':
    main()
