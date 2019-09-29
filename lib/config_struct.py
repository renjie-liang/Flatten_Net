
cfg.NUM_GPUS
cfg.TRAIN.IMS_PER_BATCH



__C
    --TRAIN
        --BATCH_SIZE = 10
        --BATCH_SIZE_PER_GPU = 10
        --EPOCH                 = 100
        --RUNS_FOLDER           = ''
        --START_EPOCH           = 1
        --NAME                  = ''
        --DATA_DIR              = ''

    --INPUT_SIZE
        --H_IMG     = 256
        --W_IMG     = 256 
        --H_WM      = 32
        --W_WM      = 32
    --HIDDEN
        --ENCCODER
            --IMG_ENCODER_CHANNELS = 64 
            --IMG_ENCODER_BLOCKS= 4
            --WM_ENCODER_CHANNELS = 32
            --WM_ENCODER_BLOCKS = 4
            --ENCODER_LOSS=10,

        --DECODER
            --DECODER_BLOCKS=7
            --DECODER_CHANNELS=64
            --DECODER_LOSS=1
        --ENABLE_FP16=ARGS.ENABLE_FP16






__C
    --TRAIN
        --DATASETS              = ()
        --DEBUG                 = False
        --BATCH_SIZE            = 10
        --BATCH_SIZE_PER_GPU    = 10


        --SCALES                = (600, )
        --MAX_SIZE              = 1000
        --IMS_PER_BATCH         = 1
        --BATCH_SIZE_PER_IM     = 64

        --FG_FRACTION           = 0.25
        --FG_THRESH             = 0.5
        --BG_THRESH_HI          = 0.5
        --BG_THRESH_LO          = 0.0
        --USE_FLIPPED           = True
        --BBOX_THRESH           = 0.5
        --PROPOSAL_FILES        = ()
        --SNAPSHOT_ITERS        = 20000
        --BBOX_NORMALIZE_TARGETS        = True
        --BBOX_INSIDE_WEIGHTS           = (1.0, 1.0, 1.0, 1.0)
        --BBOX_NORMALIZE_TARGETS_PRECOMPUTED        = False
        --BBOX_NORMALIZE_TARGETS    = True
        --BBOX_INSIDE_WEIGHTS       = (1.0, 1.0, 1.0, 1.0)
        --BBOX_NORMALIZE_TARGETS_PRECOMPUTED        = False
        --BBOX_NORMALIZE_MEANS      = (0.0, 0.0, 0.0, 0.0)
        --BBOX_NORMALIZE_STDS       = (0.1, 0.1, 0.2, 0.2)
        --ASPECT_GROUPING           = True
        --ASPECT_CROPPING           = False
        --ASPECT_HI                 = 2
        --ASPECT_LO                 = 0.5

        --RPN_POSITIVE_OVERLAP      = 0.7
        --RPN_NEGATIVE_OVERLAP      = 0.3
        --RPN_FG_FRACTION           = 0.5
        --RPN_BATCH_SIZE_PER_IM         = 256
        --RPN_NMS_THRESH            = 0.7
        --RPN_PRE_NMS_TOP_N         = 12000
        --RPN_POST_NMS_TOP_N        = 2000
        --RPN_STRADDLE_THRESH       = 0
        --RPN_MIN_SIZE              = 0
        --CROWD_FILTER_THRESH       = 0.7
        --GT_MIN_AREA               = -1
        --FREEZE_CONV_BODY          = False


    --DATA_LOADER
        --NUM_THREADS   = 4

    --TEST
        --DATASETS      = ()
        --SCALE         = 600
        --MAX_SIZE      = 1000
        --NMS           = 0.3
        --BBOX_REG      = True
        --PROPOSAL_FILES        = ()
        --PROPOSAL_LIMIT        = 2000
        --RPN_NMS_THRESH        = 0.7
        --RPN_PRE_NMS_TOP_N     = 12000
        --RPN_POST_NMS_TOP_N    = 2000
        --RPN_MIN_SIZE          = 0
        --DETECTIONS_PER_IM     = 100
        --SCORE_THRESH          = 0.05
        --COMPETITION_MODE      = True
        --FORCE_JSON_DATASET_EVAL       = False
        --PRECOMPUTED_PROPOSALS         = True

        --BBOX_AUG = AttrDict()
            --ENABLED = False
            --SCORE_HEUR = 'UNION'
            --COORD_HEUR = 'UNION'
            --H_FLIP = False
            --SCALES = ()
            --MAX_SIZE = 4000
            --SCALE_H_FLIP = False
            --SCALE_SIZE_DEP = False
            --AREA_TH_LO = 50**2
            --AREA_TH_HI = 180**2
            --ASPECT_RATIOS = ()
            --ASPECT_RATIO_H_FLIP = False

        --MASK_AUG = AttrDict()
            --ENABLED = False
            --HEUR = 'SOFT_AVG'
            --H_FLIP = False
            --SCALES = ()
            --MAX_SIZE = 4000
            --SCALE_H_FLIP = False
            --SCALE_SIZE_DEP = False
            --AREA_TH = 180**2
            --ASPECT_RATIOS = ()
            --ASPECT_RATIO_H_FLIP = False

        --KPS_AUG = AttrDict()
            --ENABLED = False
            --HEUR = 'HM_AVG'
            --H_FLIP = False
            --SCALES = ()
            --MAX_SIZE = 4000
            --SCALE_H_FLIP = False
            --SCALE_SIZE_DEP = False
            --AREA_TH = 180**2
            --ASPECT_RATIOS = ()
            --ASPECT_RATIO_H_FLIP = False

        --SOFT_NMS = AttrDict()
            --ENABLED = False
            --METHOD = 'linear'
            --SIGMA = 0.5

        --BBOX_VOTE = AttrDict()
            --ENABLED = False
            --VOTE_TH = 0.8
            --SCORING_METHOD = 'ID'
            --SCORING_METHOD_BETA = 1.0

    --SEM = AttrDict()
        --UNION = False
        --SEM_ON = True
        --INPUT_SIZE = [240, 240]
        --TRAINSET = 'train'
        --DECODER_TYPE = 'ppm_bilinear'
        --DIM = 512
        --FC_DIM = 2048
        --DOWNSAMPLE = [0]
        --DEEP_SUB_SCALE = [1.0]
        --OUTPUT_PREFIX = 'semseg_label'
        --ARCH_ENCODER = 'resnet50_dilated8'
        --USE_RESNET = False
        --DILATED = 1
        --BN_LEARN = False
        --LAYER_FIXED = False
        --CONV3D = False
        --PSPNET_PRETRAINED_WEIGHTS = ''
        --PSPNET_REQUIRES_GRAD = True
        --SD_DIM = 512
        --FPN_DIMS = [2048, 256]
        --DROPOUT_RATIO = 0.1
        --USE_GE_BLOCK=False

    --DISP = AttrDict()
        --DISP_ON = False
        --DIM = 256
        --OUTPUT_PREFIX = 'disp_label'
        --DOWNSAMPLE = [0, 1, 2, 3]
        --DEEP_SUB_SCALE = [0.2, 0.2, 0.2, 0.2, 1.0]
        --USE_DEEPSUP = False
        --USE_CRL_DISPRES  = False
        --USE_CRL_DISPFUL = False
        --ORIGINAL = True
        --USE_MULTISCALELOSS = False
        --MAX_DISPLACEMENT = 40
        --FEATURE_MAX_DISPLACEMENT = 48
        --DISPSEG_REQUIRES_GRAD = True
        --EXPECT_MAXDISP = 127
        --COST_VOLUME_TYPE = 'CorrelationLayer1D'
        --MERGE_ASPP = True
        --MAX_DISP = 192
        --SIMPLE = False

    --DEPTH = AttrDict()
        --DEPTH_ON = False
        --DIM = 256
        --OUTPUT_PREFIX = 'depth_label'
        --DOWNSAMPLE = [0, 1, 2, 3]
        --DEEP_SUB_SCALE = [0.2, 0.2, 0.2, 0.2, 1.0]
        --USE_DEEPSUP = False
        --USE_CRL_DISPRES  = False
        --USE_CRL_DISPFUL = False
        --ORIGINAL = True
        --USE_MULTISCALELOSS = False
        --MAX_DISPLACEMENT = 40
        --FEATURE_MAX_DISPLACEMENT = 48
        --DISPSEG_REQUIRES_GRAD = True
        --EXPECT_MAXDISP = 127
        --COST_VOLUME_TYPE = 'CorrelationLayer1D'
        --MERGE_ASPP = True

    --MULTASK = AttrDict()
        --PRUNE_ON = False
        --STAGES = 5
        --PRUNE_RATIO = 0.3
        --LAYER_EVEN = False
        --REINITIALIZATION = False
        --SEMSEG_ONLY = False
        --DISP_ONLY = False
        --LOSS_CONSTRAINT = 0.1
        --RETRAIN = False
        --DETACH = False
        --INTENSIFY = -1.0
        --NS_ON = False
        --BN_DECAY = 0.0001
        --FREEZE_BN_REMAIN = False
        --FREEZE_BN_ALL = False
        --INSTANT_PRUNE = False
        --ADAPTIVE_WEIGHT = False

    --REGULARIZATION = AttrDict()
        --OUR_ON = False
        --L1_NORM = False
        --L2_NORM = True
        --WEIGHT_DECAY = 0.0001
        --REG_JOINT = False
        --JOINT_DECAY = 0.0001
        --REG_ORTHO = False
        --ORTHO_DECAY = 0.0001 

    --VALIDATION = AttrDict()
        --VAL_ON = False
        --INTERVAL_EPOCH = 1
        --VAL_LIST = ''
        --START_ITER = 0
        --END_ITER = 500
        --FULL_SIZE = False

    --MODEL = AttrDict()
        --TYPE = ''
        --CONV_BODY = ''
        --NUM_CLASSES = -1
        --CLS_AGNOSTIC_BBOX_REG = False
        --BBOX_REG_WEIGHTS = (10., 10., 5., 5.)
        --FASTER_RCNN = False
        --MASK_ON = False
        --KEYPOINTS_ON = False
        --RPN_ONLY = False
        --SHARE_RES5 = False
        --LOAD_IMAGENET_PRETRAINED_WEIGHTS = True
        --UNSUPERVISED_POSE = False


    --RETINANET = AttrDict()
        --RETINANET_ON = False
        --ASPECT_RATIOS = (0.5, 1.0, 2.0)
        --SCALES_PER_OCTAVE = 3
        --ANCHOR_SCALE = 4
        --NUM_CONVS = 4
        --BBOX_REG_WEIGHT = 1.0
        --BBOX_REG_BETA = 0.11
        --PRE_NMS_TOP_N = 1000
        --POSITIVE_OVERLAP = 0.5
        --NEGATIVE_OVERLAP = 0.4
        --LOSS_ALPHA = 0.25
        --LOSS_GAMMA = 2.0
        --PRIOR_PROB = 0.01
        --SHARE_CLS_BBOX_TOWER = False
        --CLASS_SPECIFIC_BBOX = False
        --SOFTMAX = False
        --INFERENCE_TH = 0.05

    --SOLVER = AttrDict()
        --TYPE = 'SGD'
        --BASE_LR = 0.001
        --LR_POLICY = 'step'
        --GAMMA = 0.1
        --STEP_SIZE = 30000
        --STEPS = []
        --LRS = []
        --MAX_ITER = 40000
        --MOMENTUM = 0.9
        --WEIGHT_DECAY = 0.0005
        --WEIGHT_DECAY_GN = 0.0
        --BIAS_DOUBLE_LR = True
        --BIAS_WEIGHT_DECAY = False
        --WARM_UP_ITERS = 500
        --WARM_UP_FACTOR = 1.0 / 3.0
        --WARM_UP_METHOD = 'linear'
        --SCALE_MOMENTUM = True
        --SCALE_MOMENTUM_THRESHOLD = 1.1
        --LOG_LR_CHANGE_THRESHOLD = 1.1

    --FAST_RCNN = AttrDict()
        --ROI_BOX_HEAD = ''
        --MLP_HEAD_DIM = 1024
        --CONV_HEAD_DIM = 256
        --NUM_STACKED_CONVS = 4
        --ROI_XFORM_METHOD = 'RoIPoolF'
        --ROI_XFORM_SAMPLING_RATIO = 0
        --ROI_XFORM_RESOLUTION = 14

    --RPN = AttrDict()
        --RPN_ON = False
        --OUT_DIM_AS_IN_DIM = True
        --OUT_DIM = 512
        --CLS_ACTIVATION = 'sigmoid'
        --SIZES = (64, 128, 256, 512)
        --STRIDE = 16
        --ASPECT_RATIOS = (0.5, 1, 2)

    --FPN = AttrDict()
        --FPN_ON = False
        --DIM = 256
        --ZERO_INIT_LATERAL = False
        --COARSEST_STRIDE = 32
        --MULTILEVEL_ROIS = False
        --ROI_CANONICAL_SCALE = 224  # s0
        --ROI_CANONICAL_LEVEL = 4  # k0: where s0 maps to
        --ROI_MAX_LEVEL = 5
        --ROI_MIN_LEVEL = 2

        --MULTILEVEL_RPN = False
        --RPN_MAX_LEVEL = 6
        --RPN_MIN_LEVEL = 2
        --RPN_ASPECT_RATIOS = (0.5, 1, 2)
        --RPN_ANCHOR_START_SIZE = 32
        --RPN_COLLECT_SCALE = 1
        --EXTRA_CONV_LEVELS = False
        --USE_GN = False

    --MRCNN = AttrDict()
        --ROI_MASK_HEAD = ''
        --RESOLUTION = 14
        --ROI_XFORM_METHOD = 'RoIAlign'
        --ROI_XFORM_RESOLUTION = 7
        --ROI_XFORM_SAMPLING_RATIO = 0
        --DIM_REDUCED = 256
        --DILATION = 2
        --UPSAMPLE_RATIO = 1
        --USE_FC_OUTPUT = False
        --CONV_INIT = 'GaussianFill'
        --CLS_SPECIFIC_MASK = True
        --WEIGHT_LOSS_MASK = 1.0
        --THRESH_BINARIZE = 0.5
        --MEMORY_EFFICIENT_LOSS = True  # TODO



    --KRCNN = AttrDict()
        --ROI_KEYPOINTS_HEAD = ''
        --HEATMAP_SIZE = -1
        --UP_SCALE = -1
        --USE_DECONV = False
        --DECONV_DIM = 256
        --USE_DECONV_OUTPUT = False
        --DILATION = 1
        --DECONV_KERNEL = 4
        --NUM_KEYPOINTS = -1
        --NUM_STACKED_CONVS = 8
        --CONV_HEAD_DIM = 256
        --CONV_HEAD_KERNEL = 3
        --CONV_INIT = 'GaussianFill'
        --NMS_OKS = False
        --KEYPOINT_CONFIDENCE = 'bbox'
        --ROI_XFORM_METHOD = 'RoIAlign'
        --ROI_XFORM_RESOLUTION = 7
        --ROI_XFORM_SAMPLING_RATIO = 0
        --MIN_KEYPOINT_COUNT_FOR_VALID_MINIBATCH = 20
        --INFERENCE_MIN_SIZE = 0
        --LOSS_WEIGHT = 1.0
        --NORMALIZE_BY_VISIBLE_KEYPOINTS = True

    --RFCN = AttrDict()
        --PS_GRID_SIZE = 3

    --RESNETS = AttrDict()
        --NUM_GROUPS = 1
        --WIDTH_PER_GROUP = 64
        --STRIDE_1X1 = True
        --TRANS_FUNC = 'bottleneck_transformation'
        --STEM_FUNC = 'basic_bn_stem'
        --SHORTCUT_FUNC = 'basic_bn_shortcut'
        --RES5_DILATION = 1
        --FREEZE_AT = 2
        --IMAGENET_PRETRAINED_WEIGHTS = ''
        --USE_GN = False


    --GROUP_NORM = AttrDict()
        --DIM_PER_GP = -1
        --NUM_GROUPS = 32
        --EPSILON = 1e-5

    --NUM_GPUS = 1
    --DEDUP_BOXES = 1. / 16.
    --BBOX_XFORM_CLIP = np.log(1000. / 16.)
    --PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
    --RNG_SEED = 3
    --EPS = 1e-14
    --ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', 'datasets'))
    --OUTPUT_DIR = 'Outputs'
    --MATLAB = 'matlab'
    --VIS = False
    --VIS_TH = 0.9
    --EXPECTED_RESULTS = []
    --EXPECTED_RESULTS_RTOL = 0.1
    --EXPECTED_RESULTS_ATOL = 0.005
    --EXPECTED_RESULTS_EMAIL = ''
    --DATA_DIR = osp.abspath(osp.join(    --ROOT_DIR, 'data'))
    --POOLING_MODE = 'crop'
    --POOLING_SIZE = 7
    --CROP_RESIZE_WITH_MAX_POOL = True
    --CUDA = False
    --DEBUG = False
    --PYTORCH_VERSION_LESS_THAN_040 = False
