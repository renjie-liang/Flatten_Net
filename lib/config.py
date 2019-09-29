from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import six
import os
import os.path as osp
import copy
from ast import literal_eval

import numpy as np
from packaging import version
import torch
import torch.nn as nn
from torch.nn import init
import yaml

import  torch.nn as mynn
from lib.utilis.collections import AttrDict


__C = AttrDict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C


# Random note: avoid using '.ON' as a config key since yaml converts it to True;
# prefer 'ENABLED' instead

# ---------------------------------------------------------------------------- #
# Training options
# ---------------------------------------------------------------------------- #
__C.DATA_SET = AttrDict()
__C.DATA_SET.H_IMG = 32
__C.DATA_SET.W_IMG = 32
__C.DATA_SET.H_WM = 2
__C.DATA_SET.W_WM = 2
__C.DATA_SET.INPUT_SIZE   = 5
__C.DATA_SET.OUTPUT_SIZE = 2

__C.DATA_SET.SAVE_IMGS = 4
__C.DATA_SET.SAVE_IMGS_SIZE = 256
__C.DATA_SET.SAVE_IMGS_DIR = ''

__C.DATA_SET.SAVE_WMS = 4
__C.DATA_SET.SAVE_WMS_SCALES = 8
__C.DATA_SET.SAVE_WMS_DIR = ''



__C.DATA_LOADER = AttrDict()
__C.DATA_LOADER.DATA_DIR = ''
__C.DATA_LOADER.TRAIN_IMG_FOLDER = ''
__C.DATA_LOADER.VAL_IMG_FOLDER = ''
__C.DATA_LOADER.NUM_THREADS = 4

__C.MODEL = AttrDict()
__C.MODEL.NAME = ''
__C.MODEL.ENCODER_NAME  = ''
__C.MODEL.DECODER_NAME  = ''
__C.MODEL.NOSIER_NAME   = ''
__C.MODEL.NET_NAME   = ''
__C.MODEL.IMG_LOSS_NAME = ''
__C.MODEL.WM_LOSS_NAME = ''
__C.MODEL.LOSS_NAME = ''
__C.MODEL.ENCODER_LOSS=0.7
__C.MODEL.DECODER_LOSS=10

__C.Noiser = AttrDict()
__C.Noiser.SET_LIST = ''
__C.Noiser.RANDOM_NUM = None

__C.HiddenNet = AttrDict()
__C.HiddenNet.IMG_ENCODER_CHANNELS = 64 
__C.HiddenNet.IMG_ENCODER_BLOCKS= 4
__C.HiddenNet.ENABLE_FP16 = False

__C.BASE_Decoder = AttrDict()
__C.BASE_Decoder.DECODER_CHANNELS=64
__C.BASE_Decoder.DECODER_BLOCKS=7


__C.DeConv = AttrDict()
__C.DeConv.BASE = AttrDict()
__C.DeConv.BASE.SIGMOID =  True

__C.DeConv.WMFlatten = AttrDict()
__C.DeConv.WMFlatten.KERNEL_SIZE = 1

__C.DeConv.WMConv_ENCODER = AttrDict()
__C.DeConv.WMConv_ENCODER.WM_LAYERS_NUM = 0

__C.DeConv.WMConv_DECODER = AttrDict()
__C.DeConv.WMConv_DECODER.WM_LAYERS_NUM = 0

__C.TRAIN = AttrDict()
__C.TRAIN.DEBUG     = False
__C.TRAIN.NO_SAVE    = False  

__C.TRAIN.BATCH_SIZE = 10
__C.TRAIN.BATCH_SIZE_PER_GPU = 10
__C.TRAIN.EPOCHS                = 5
__C.TRAIN.RUNS_FOLDER           = ''
__C.TRAIN.START_EPOCH           = 1
__C.TRAIN.NAME                  = ''
__C.TRAIN.DATA_DIR              = ''
__C.TRAIN.DEVICE = ''
__C.TRAIN.NUM_GPUS = 1
__C.TRAIN.PRINT_STEP = 100
__C.TRAIN.STEP_PER_EPOCH = None

__C.VAL = AttrDict()
__C.VAL.STEP_PER_EPOCH = 0




__C.TEST = AttrDict()
__C.TEST.DATASETS = ()
__C.TEST.SCALE = 600
__C.TEST.MAX_SIZE = 1000
__C.TEST.NMS = 0.3
__C.TEST.BBOX_REG = True
__C.TEST.PROPOSAL_FILES = ()
__C.TEST.PROPOSAL_LIMIT = 2000
__C.TEST.RPN_NMS_THRESH = 0.7
__C.TEST.RPN_PRE_NMS_TOP_N = 12000
__C.TEST.RPN_POST_NMS_TOP_N = 2000
__C.TEST.RPN_MIN_SIZE = 0
__C.TEST.DETECTIONS_PER_IM = 100
__C.TEST.SCORE_THRESH = 0.05
__C.TEST.COMPETITION_MODE = True
__C.TEST.FORCE_JSON_DATASET_EVAL = False
__C.TEST.PRECOMPUTED_PROPOSALS = True
__C.TEST.BBOX_AUG = AttrDict()
__C.TEST.BBOX_AUG.ENABLED = False
__C.TEST.BBOX_AUG.SCORE_HEUR = 'UNION'
__C.TEST.BBOX_AUG.COORD_HEUR = 'UNION'
__C.TEST.BBOX_AUG.H_FLIP = False
__C.TEST.BBOX_AUG.SCALES = ()
__C.TEST.BBOX_AUG.MAX_SIZE = 4000
__C.TEST.BBOX_AUG.SCALE_H_FLIP = False
__C.TEST.BBOX_AUG.SCALE_SIZE_DEP = False
__C.TEST.BBOX_AUG.AREA_TH_LO = 50**2
__C.TEST.BBOX_AUG.AREA_TH_HI = 180**2
__C.TEST.BBOX_AUG.ASPECT_RATIOS = ()
__C.TEST.BBOX_AUG.ASPECT_RATIO_H_FLIP = False
__C.TEST.MASK_AUG = AttrDict()
__C.TEST.MASK_AUG.ENABLED = False
__C.TEST.MASK_AUG.HEUR = 'SOFT_AVG'
__C.TEST.MASK_AUG.H_FLIP = False
__C.TEST.MASK_AUG.SCALES = ()
__C.TEST.MASK_AUG.MAX_SIZE = 4000
__C.TEST.MASK_AUG.SCALE_H_FLIP = False
__C.TEST.MASK_AUG.SCALE_SIZE_DEP = False
__C.TEST.MASK_AUG.AREA_TH = 180**2
__C.TEST.MASK_AUG.ASPECT_RATIOS = ()
__C.TEST.MASK_AUG.ASPECT_RATIO_H_FLIP = False
__C.TEST.KPS_AUG = AttrDict()
__C.TEST.KPS_AUG.ENABLED = False
__C.TEST.KPS_AUG.HEUR = 'HM_AVG'
__C.TEST.KPS_AUG.H_FLIP = False
__C.TEST.KPS_AUG.SCALES = ()
__C.TEST.KPS_AUG.MAX_SIZE = 4000
__C.TEST.KPS_AUG.SCALE_H_FLIP = False
__C.TEST.KPS_AUG.SCALE_SIZE_DEP = False
__C.TEST.KPS_AUG.AREA_TH = 180**2
__C.TEST.KPS_AUG.ASPECT_RATIOS = ()
__C.TEST.KPS_AUG.ASPECT_RATIO_H_FLIP = False
__C.TEST.SOFT_NMS = AttrDict()
__C.TEST.SOFT_NMS.ENABLED = False
__C.TEST.SOFT_NMS.METHOD = 'linear'
__C.TEST.SOFT_NMS.SIGMA = 0.5
__C.TEST.BBOX_VOTE = AttrDict()
__C.TEST.BBOX_VOTE.ENABLED = False
__C.TEST.BBOX_VOTE.VOTE_TH = 0.8
__C.TEST.BBOX_VOTE.SCORING_METHOD = 'ID'
__C.TEST.BBOX_VOTE.SCORING_METHOD_BETA = 1.0

__C.SEM = AttrDict()
__C.SEM.UNION = False
__C.SEM.SEM_ON = True
__C.SEM.INPUT_SIZE = [240, 240]
__C.SEM.TRAINSET = 'train'
__C.SEM.DECODER_TYPE = 'ppm_bilinear'
__C.SEM.DIM = 512
__C.SEM.FC_DIM = 2048
__C.SEM.DOWNSAMPLE = [0]
__C.SEM.DEEP_SUB_SCALE = [1.0]
__C.SEM.OUTPUT_PREFIX = 'semseg_label'
__C.SEM.ARCH_ENCODER = 'resnet50_dilated8'
__C.SEM.USE_RESNET = False
__C.SEM.DILATED = 1
__C.SEM.BN_LEARN = False
__C.SEM.LAYER_FIXED = False
__C.SEM.CONV3D = False
__C.SEM.PSPNET_PRETRAINED_WEIGHTS = ''
__C.SEM.PSPNET_REQUIRES_GRAD = True
__C.SEM.SD_DIM = 512
__C.SEM.FPN_DIMS = [2048, 256]
__C.SEM.DROPOUT_RATIO = 0.1
__C.SEM.USE_GE_BLOCK=False


__C.DISP = AttrDict()
__C.DISP.DISP_ON = False
__C.DISP.DIM = 256
__C.DISP.OUTPUT_PREFIX = 'disp_label'
__C.DISP.DOWNSAMPLE = [0, 1, 2, 3]
__C.DISP.DEEP_SUB_SCALE = [0.2, 0.2, 0.2, 0.2, 1.0]
__C.DISP.USE_DEEPSUP = False
__C.DISP.USE_CRL_DISPRES  = False
__C.DISP.USE_CRL_DISPFUL = False
__C.DISP.ORIGINAL = True
__C.DISP.USE_MULTISCALELOSS = False
__C.DISP.MAX_DISPLACEMENT = 40
__C.DISP.FEATURE_MAX_DISPLACEMENT = 48
__C.DISP.DISPSEG_REQUIRES_GRAD = True
__C.DISP.EXPECT_MAXDISP = 127
__C.DISP.COST_VOLUME_TYPE = 'CorrelationLayer1D'
__C.DISP.MERGE_ASPP = True
__C.DISP.MAX_DISP = 192
__C.DISP.SIMPLE = False

__C.DEPTH = AttrDict()
__C.DEPTH.DEPTH_ON = False
__C.DEPTH.DIM = 256
__C.DEPTH.OUTPUT_PREFIX = 'depth_label'
__C.DEPTH.DOWNSAMPLE = [0, 1, 2, 3]
__C.DEPTH.DEEP_SUB_SCALE = [0.2, 0.2, 0.2, 0.2, 1.0]
__C.DEPTH.USE_DEEPSUP = False
__C.DEPTH.USE_CRL_DISPRES  = False
__C.DEPTH.USE_CRL_DISPFUL = False
__C.DEPTH.ORIGINAL = True
__C.DEPTH.USE_MULTISCALELOSS = False
__C.DEPTH.MAX_DISPLACEMENT = 40
__C.DEPTH.FEATURE_MAX_DISPLACEMENT = 48
__C.DEPTH.DISPSEG_REQUIRES_GRAD = True
__C.DEPTH.EXPECT_MAXDISP = 127
__C.DEPTH.COST_VOLUME_TYPE = 'CorrelationLayer1D'
__C.DEPTH.MERGE_ASPP = True

__C.MULTASK = AttrDict()
__C.MULTASK.PRUNE_ON = False
__C.MULTASK.STAGES = 5
__C.MULTASK.PRUNE_RATIO = 0.3
__C.MULTASK.LAYER_EVEN = False
__C.MULTASK.REINITIALIZATION = False
__C.MULTASK.SEMSEG_ONLY = False
__C.MULTASK.DISP_ONLY = False
__C.MULTASK.LOSS_CONSTRAINT = 0.1
__C.MULTASK.RETRAIN = False
__C.MULTASK.DETACH = False
__C.MULTASK.INTENSIFY = -1.0
__C.MULTASK.NS_ON = False
__C.MULTASK.BN_DECAY = 0.0001
__C.MULTASK.FREEZE_BN_REMAIN = False
__C.MULTASK.FREEZE_BN_ALL = False
__C.MULTASK.INSTANT_PRUNE = False
__C.MULTASK.ADAPTIVE_WEIGHT = False

__C.REGULARIZATION = AttrDict()
__C.REGULARIZATION.OUR_ON = False
__C.REGULARIZATION.L1_NORM = False
__C.REGULARIZATION.L2_NORM = True
__C.REGULARIZATION.WEIGHT_DECAY = 0.0001
__C.REGULARIZATION.REG_JOINT = False
__C.REGULARIZATION.JOINT_DECAY = 0.0001
__C.REGULARIZATION.REG_ORTHO = False
__C.REGULARIZATION.ORTHO_DECAY = 0.0001 

__C.VALIDATION = AttrDict()
__C.VALIDATION.VAL_ON = False
__C.VALIDATION.INTERVAL_EPOCH = 1
__C.VALIDATION.VAL_LIST = ''
__C.VALIDATION.START_ITER = 0
__C.VALIDATION.END_ITER = 500
__C.VALIDATION.FULL_SIZE = False




__C.RETINANET = AttrDict()
__C.RETINANET.RETINANET_ON = False
__C.RETINANET.ASPECT_RATIOS = (0.5, 1.0, 2.0)
__C.RETINANET.SCALES_PER_OCTAVE = 3
__C.RETINANET.ANCHOR_SCALE = 4
__C.RETINANET.NUM_CONVS = 4
__C.RETINANET.BBOX_REG_WEIGHT = 1.0
__C.RETINANET.BBOX_REG_BETA = 0.11
__C.RETINANET.PRE_NMS_TOP_N = 1000
__C.RETINANET.POSITIVE_OVERLAP = 0.5
__C.RETINANET.NEGATIVE_OVERLAP = 0.4
__C.RETINANET.LOSS_ALPHA = 0.25
__C.RETINANET.LOSS_GAMMA = 2.0
__C.RETINANET.PRIOR_PROB = 0.01
__C.RETINANET.SHARE_CLS_BBOX_TOWER = False
__C.RETINANET.CLASS_SPECIFIC_BBOX = False
__C.RETINANET.SOFTMAX = False
__C.RETINANET.INFERENCE_TH = 0.05

__C.SOLVER = AttrDict()
__C.SOLVER.TYPE = 'SGD'
__C.SOLVER.BASE_LR = 0.001
__C.SOLVER.LR_POLICY = 'step'
__C.SOLVER.GAMMA = 0.1
__C.SOLVER.STEP_SIZE = 30000
__C.SOLVER.STEPS = []
__C.SOLVER.LRS = []
__C.SOLVER.MAX_ITER = 40000
__C.SOLVER.MOMENTUM = 0.9
__C.SOLVER.WEIGHT_DECAY = 0.0005
__C.SOLVER.WEIGHT_DECAY_GN = 0.0
__C.SOLVER.BIAS_DOUBLE_LR = True
__C.SOLVER.BIAS_WEIGHT_DECAY = False
__C.SOLVER.WARM_UP_ITERS = 500
__C.SOLVER.WARM_UP_FACTOR = 1.0 / 3.0
__C.SOLVER.WARM_UP_METHOD = 'linear'
__C.SOLVER.SCALE_MOMENTUM = True
__C.SOLVER.SCALE_MOMENTUM_THRESHOLD = 1.1
__C.SOLVER.LOG_LR_CHANGE_THRESHOLD = 1.1

__C.FAST_RCNN = AttrDict()
__C.FAST_RCNN.ROI_BOX_HEAD = ''
__C.FAST_RCNN.MLP_HEAD_DIM = 1024
__C.FAST_RCNN.CONV_HEAD_DIM = 256
__C.FAST_RCNN.NUM_STACKED_CONVS = 4
__C.FAST_RCNN.ROI_XFORM_METHOD = 'RoIPoolF'
__C.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO = 0
__C.FAST_RCNN.ROI_XFORM_RESOLUTION = 14

__C.RPN = AttrDict()
__C.RPN.RPN_ON = False
__C.RPN.OUT_DIM_AS_IN_DIM = True
__C.RPN.OUT_DIM = 512
__C.RPN.CLS_ACTIVATION = 'sigmoid'
__C.RPN.SIZES = (64, 128, 256, 512)
__C.RPN.STRIDE = 16
__C.RPN.ASPECT_RATIOS = (0.5, 1, 2)

__C.FPN = AttrDict()
__C.FPN.FPN_ON = False
__C.FPN.DIM = 256
__C.FPN.ZERO_INIT_LATERAL = False
__C.FPN.COARSEST_STRIDE = 32
__C.FPN.MULTILEVEL_ROIS = False
__C.FPN.ROI_CANONICAL_SCALE = 224  # s0
__C.FPN.ROI_CANONICAL_LEVEL = 4  # k0: where s0 maps to
__C.FPN.ROI_MAX_LEVEL = 5
__C.FPN.ROI_MIN_LEVEL = 2
__C.FPN.MULTILEVEL_RPN = False
__C.FPN.RPN_MAX_LEVEL = 6
__C.FPN.RPN_MIN_LEVEL = 2
__C.FPN.RPN_ASPECT_RATIOS = (0.5, 1, 2)
__C.FPN.RPN_ANCHOR_START_SIZE = 32
__C.FPN.RPN_COLLECT_SCALE = 1
__C.FPN.EXTRA_CONV_LEVELS = False
__C.FPN.USE_GN = False

__C.MRCNN = AttrDict()
__C.MRCNN.ROI_MASK_HEAD = ''
__C.MRCNN.RESOLUTION = 14
__C.MRCNN.ROI_XFORM_METHOD = 'RoIAlign'
__C.MRCNN.ROI_XFORM_RESOLUTION = 7
__C.MRCNN.ROI_XFORM_SAMPLING_RATIO = 0
__C.MRCNN.DIM_REDUCED = 256
__C.MRCNN.DILATION = 2
__C.MRCNN.UPSAMPLE_RATIO = 1
__C.MRCNN.USE_FC_OUTPUT = False
__C.MRCNN.CONV_INIT = 'GaussianFill'
__C.MRCNN.CLS_SPECIFIC_MASK = True
__C.MRCNN.WEIGHT_LOSS_MASK = 1.0
__C.MRCNN.THRESH_BINARIZE = 0.5
__C.MRCNN.MEMORY_EFFICIENT_LOSS = True  # TODO

__C.KRCNN = AttrDict()
__C.KRCNN.ROI_KEYPOINTS_HEAD = ''
__C.KRCNN.HEATMAP_SIZE = -1
__C.KRCNN.UP_SCALE = -1
__C.KRCNN.USE_DECONV = False
__C.KRCNN.DECONV_DIM = 256
__C.KRCNN.USE_DECONV_OUTPUT = False
__C.KRCNN.DILATION = 1
__C.KRCNN.DECONV_KERNEL = 4
__C.KRCNN.NUM_KEYPOINTS = -1
__C.KRCNN.NUM_STACKED_CONVS = 8
__C.KRCNN.CONV_HEAD_DIM = 256
__C.KRCNN.CONV_HEAD_KERNEL = 3
__C.KRCNN.CONV_INIT = 'GaussianFill'
__C.KRCNN.NMS_OKS = False
__C.KRCNN.KEYPOINT_CONFIDENCE = 'bbox'
__C.KRCNN.ROI_XFORM_METHOD = 'RoIAlign'
__C.KRCNN.ROI_XFORM_RESOLUTION = 7
__C.KRCNN.ROI_XFORM_SAMPLING_RATIO = 0
__C.KRCNN.MIN_KEYPOINT_COUNT_FOR_VALID_MINIBATCH = 20
__C.KRCNN.INFERENCE_MIN_SIZE = 0
__C.KRCNN.LOSS_WEIGHT = 1.0
__C.KRCNN.NORMALIZE_BY_VISIBLE_KEYPOINTS = True

__C.RFCN = AttrDict()
__C.RFCN.PS_GRID_SIZE = 3

__C.RESNETS = AttrDict()
__C.RESNETS.NUM_GROUPS = 1
__C.RESNETS.WIDTH_PER_GROUP = 64
__C.RESNETS.STRIDE_1X1 = True
__C.RESNETS.TRANS_FUNC = 'bottleneck_transformation'
__C.RESNETS.STEM_FUNC = 'basic_bn_stem'
__C.RESNETS.SHORTCUT_FUNC = 'basic_bn_shortcut'
__C.RESNETS.RES5_DILATION = 1
__C.RESNETS.FREEZE_AT = 2
__C.RESNETS.IMAGENET_PRETRAINED_WEIGHTS = ''
__C.RESNETS.USE_GN = False

__C.GROUP_NORM = AttrDict()
__C.GROUP_NORM.DIM_PER_GP = -1
__C.GROUP_NORM.NUM_GROUPS = 32
__C.GROUP_NORM.EPSILON = 1e-5

__C.NUM_GPUS = 1
__C.DEDUP_BOXES = 1. / 16.
__C.BBOX_XFORM_CLIP = np.log(1000. / 16.)
__C.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
__C.RNG_SEED = 3
__C.EPS = 1e-14
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', 'datasets'))
__C.OUTPUT_DIR = 'Outputs'
__C.MATLAB = 'matlab'
__C.VIS = False
__C.VIS_TH = 0.9
__C.EXPECTED_RESULTS = []
__C.EXPECTED_RESULTS_RTOL = 0.1
__C.EXPECTED_RESULTS_ATOL = 0.005
__C.EXPECTED_RESULTS_EMAIL = ''

__C.DATA_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'data'))
__C.POOLING_MODE = 'crop'
__C.POOLING_SIZE = 7
__C.CROP_RESIZE_WITH_MAX_POOL = True
__C.CUDA = False
__C.DEBUG = False
__C.PYTORCH_VERSION_LESS_THAN_040 = False


# ---------------------------------------------------------------------------- #
# mask heads or keypoint heads that share res5 stage weights and
# training forward computation with box head.
# ---------------------------------------------------------------------------- #
_SHARE_RES5_HEADS = set(
    [
        'mask_rcnn_heads.mask_rcnn_fcn_head_v0upshare',
    ]
)


def assert_and_infer_cfg(make_immutable=True):
    """Call this function in your script after you have finished setting all cfg
    values that are necessary (e.g., merging a config from a file, merging
    command line config options, etc.). By default, this function will also
    mark the global cfg as immutable to prevent changing the global cfg settings
    during script execution (which can lead to hard to debug errors or code
    that's harder to understand than is necessary).
    """
    if __C.MODEL.RPN_ONLY or __C.MODEL.FASTER_RCNN:
        __C.RPN.RPN_ON = True
    if __C.RPN.RPN_ON or __C.RETINANET.RETINANET_ON:
        __C.TEST.PRECOMPUTED_PROPOSALS = False
    if __C.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS:
        assert __C.RESNETS.IMAGENET_PRETRAINED_WEIGHTS, \
            "Path to the weight file must not be empty to load imagenet pertrained resnets."
    if set([__C.MRCNN.ROI_MASK_HEAD, __C.KRCNN.ROI_KEYPOINTS_HEAD]) & _SHARE_RES5_HEADS:
        __C.MODEL.SHARE_RES5 = True
    if version.parse(torch.__version__) < version.parse('0.4.0'):
        __C.PYTORCH_VERSION_LESS_THAN_040 = True
        # create alias for PyTorch version less than 0.4.0
        init.uniform_ = init.uniform
        init.normal_ = init.normal
        init.constant_ = init.constant
        nn.GroupNorm = mynn.GroupNorm
    if make_immutable:
        cfg.immutable(True)


def merge_cfg_from_file(cfg_filename):
    """Load a yaml config file and merge it into the global config."""
    with open(cfg_filename, 'r') as f:

        yaml_cfg = AttrDict(yaml.load(f, Loader=yaml.FullLoader))
    _merge_a_into_b(yaml_cfg, __C)

cfg_from_file = merge_cfg_from_file


def merge_cfg_from_cfg(cfg_other):
    """Merge `cfg_other` into the global config."""
    _merge_a_into_b(cfg_other, __C)


def merge_cfg_from_list(cfg_list):
    """Merge config keys, values in a list (e.g., from command line) into the
    global config. For example, `cfg_list = ['TEST.NMS', 0.5]`.
    """
    assert len(cfg_list) % 2 == 0
    for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):
        # if _key_is_deprecated(full_key):
        #     continue
        # if _key_is_renamed(full_key):
        #     _raise_key_rename_error(full_key)
        key_list = full_key.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert subkey in d, 'Non-existent key: {}'.format(full_key)
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d, 'Non-existent key: {}'.format(full_key)
        value = _decode_cfg_value(v)
        value = _check_and_coerce_cfg_value_type(
            value, d[subkey], subkey, full_key
        )
        d[subkey] = value

cfg_from_list = merge_cfg_from_list


def _merge_a_into_b(a, b, stack=None):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    assert isinstance(a, AttrDict), 'Argument `a` must be an AttrDict'
    assert isinstance(b, AttrDict), 'Argument `b` must be an AttrDict'

    for k, v_ in a.items():
        full_key = '.'.join(stack) + '.' + k if stack is not None else k
        # a must specify keys that are in b
        if k not in b:
            # if _key_is_deprecated(full_key):
            #     continue
            # elif _key_is_renamed(full_key):
            #     _raise_key_rename_error(full_key)
            # else:
            raise KeyError('Non-existent config key: {}'.format(full_key))

        v = copy.deepcopy(v_)
        v = _decode_cfg_value(v)
        v = _check_and_coerce_cfg_value_type(v, b[k], k, full_key)

        # Recursively merge dicts
        if isinstance(v, AttrDict):
            try:
                stack_push = [k] if stack is None else stack + [k]
                _merge_a_into_b(v, b[k], stack=stack_push)
            except BaseException:
                raise
        else:
            b[k] = v


def _decode_cfg_value(v):
    """Decodes a raw config value (e.g., from a yaml config files or command
    line argument) into a Python object.
    """
    # Configs parsed from raw yaml will contain dictionary keys that need to be
    # converted to AttrDict objects
    if isinstance(v, dict):
        return AttrDict(v)
    # All remaining processing is only applied to strings
    if not isinstance(v, six.string_types):
        return v
    # Try to interpret `v` as a:
    #   string, number, tuple, list, dict, boolean, or None
    try:
        v = literal_eval(v)
    # The following two excepts allow v to pass through when it represents a
    # string.
    #
    # Longer explanation:
    # The type of v is always a string (before calling literal_eval), but
    # sometimes it *represents* a string and other times a data structure, like
    # a list. In the case that v represents a string, what we got back from the
    # yaml parser is 'foo' *without quotes* (so, not '"foo"'). literal_eval is
    # ok with '"foo"', but will raise a ValueError if given 'foo'. In other
    # cases, like paths (v = 'foo/bar' and not v = '"foo/bar"'), literal_eval
    # will raise a SyntaxError.
    except ValueError:
        pass
    except SyntaxError:
        pass
    return v


def _check_and_coerce_cfg_value_type(value_a, value_b, key, full_key):
    """Checks that `value_a`, which is intended to replace `value_b` is of the
    right type. The type is correct if it matches exactly or is one of a few
    cases in which the type can be easily coerced.
    """
    # The types must match (with some exceptions)
    type_b = type(value_b)
    type_a = type(value_a)
    if type_a is type_b:
        return value_a

    # Exceptions: numpy arrays, strings, tuple<->list
    if isinstance(value_b, np.ndarray):
        value_a = np.array(value_a, dtype=value_b.dtype)
    elif isinstance(value_b, six.string_types):
        value_a = str(value_a)
    elif isinstance(value_a, tuple) and isinstance(value_b, list):
        value_a = list(value_a)
    elif isinstance(value_a, list) and isinstance(value_b, tuple):
        value_a = tuple(value_a)
    else:
        raise ValueError(
            'Type mismatch ({} vs. {}) with values ({} vs. {}) for config '
            'key: {}'.format(type_b, type_a, value_b, value_a, full_key)
        )
    return value_a




def cfg_from_args(args):
    cfg.MODEL.NAME = cfg.MODEL.ENCODER_NAME + '_' + cfg.MODEL.DECODER_NAME + '_' + cfg.MODEL.IMG_LOSS_NAME +cfg.MODEL.WM_LOSS_NAME
    cfg.TRAIN.NAME = args.name
    if not cfg.DATA_LOADER.DATA_DIR:
        raise ValueError('NOT HAVE DATA_DIR')
    else:
        cfg.DATA_LOADER.TRAIN_IMG_FOLDER = os.path.join(cfg.DATA_LOADER.DATA_DIR, 'train')
        cfg.DATA_LOADER.VAL_IMG_FOLDER = os.path.join(cfg.DATA_LOADER.DATA_DIR, 'val')
    cfg.TRAIN.RUNS_FOLDER =get_run_folder_name(os.path.join('.', 'runs'), cfg.TRAIN.NAME)

    ### Adaptively adjust some configs ###

    if args.optimizer is not None:
        cfg.SOLVER.TYPE = args.optimizer
        #SGD
    if args.lr is not None:
        cfg.SOLVER.BASE_LR = args.lr
        
    if args.lr_decay_gamma is not None:
        cfg.SOLVER.GAMMA = args.lr_decay_gamma


    old_batch_size = cfg.TRAIN.BATCH_SIZE
    cfg.TRAIN.BATCH_SIZE = args.batch_size
    cfg.TRAIN.NUM_GPUS =  torch.cuda.device_count()

    assert (cfg.TRAIN.BATCH_SIZE % cfg.TRAIN.NUM_GPUS) == 0, \
        'batch_size: %d, TRAIN.NUM_GPUS: %d' % (cfg.TRAIN.BATCH_SIZE, cfg.TRAIN.NUM_GPUS)

    cfg.TRAIN.BATCH_SIZE_PER_GPU = cfg.TRAIN.BATCH_SIZE // cfg.TRAIN.NUM_GPUS

    if args.num_workers is not None:
        cfg.DATA_LOADER.NUM_THREADS = args.num_workers


    ### Adjust learning based on batch size change linearly
    old_base_lr = cfg.SOLVER.BASE_LR
    cfg.SOLVER.BASE_LR *= cfg.TRAIN.BATCH_SIZE / old_batch_size

    if not args.epochs:
        cfg.TRAIN.EPOCHS =args.epochs

    # save cfg
    if args.debug:
        cfg.TRAIN.NO_SAVE = True
        cfg.DATA_SET.H_IMG = 32
        cfg.DATA_SET.W_IMG = 32
        cfg.DATA_SET.H_WM = 4
        cfg.DATA_SET.W_WM = 4
        cfg.TRAIN.EPOCHS = 2

        # logger.info("debug: not save log, config, args and checkpoint")

        # cfg.SEM.INPUT_SIZE = [256, 256]
        # args.no_save = True
        # args.batch_size = 2
        # train_size = 8
        # args.num_epochs = 30
        # cfg.TRAIN.DATASETS = ('steel_train_on_val')

    # save log, config and args
import os
import time
def get_run_folder_name(runs_folder, experiment_name):
    """ A unique name for each run """
 
    this_run_folder = os.path.join(runs_folder, f'{experiment_name} {time.strftime("%Y.%m.%d-%H:%M:%S")}')

    return this_run_folder
    # print('Adjust BATCH_SIZE : {} --> {}'.format(
    #     old_batch_size, cfg.TRAIN.BATCH_SIZE))
    # print('Adjust BASE_LR : {} --> {}'.format(
    #     old_base_lr, cfg.SOLVER.BASE_LR))


    # print('NUM_GPUs: {}, TRAIN.BATCH_SIZE_PER_GPU: {}'.format(
    #     cfg.NUM_GPUS, cfg.TRAIN.BATCH_SIZE_PER_GPU))
    # print('TRAIN.BATCH_SIZE: {}, NUM_THREADS: {}'.format(
    #     cfg.TRAIN.BATCH_SIZE, cfg.DATA_LOADER.NUM_THREADS))