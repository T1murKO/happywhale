class config:
    SEED = 42
    
    DATA_PATH = '/root/kaggle/happywhale/data/seg_train.beton'
    SAVE_PATH = '/root/kaggle/effnetv2_m'
    
    EMBED_DIM = 512
    CLASS_NUM = 15587
    EPOCHES_NUM = 22
    BATCH_SIZE = 32
    
    IS_RESUME = True
    LOAD_PATH = '/root/kaggle/effnetv2_m'
    
    START_EPOCH = 21
    
    BACKBONE_NAME = 'effnetv2_m'
    BACKBONE_PARAMS = {'pretrained': True}
    
    TRIPLET_MARGIN = 0.3
    
    HEAD_NAME = 'adacos'
    HEAD_PARAMS = {'fixed_scale': False,
                    'feat_dim': EMBED_DIM,
                    'num_classes': CLASS_NUM}
    
    POOLING_NAME = 'gem'
    POOLING_PARAMS = {}
    
    SCHEDULER_NAME = 'decay'
    LR_SCHEDULER_PARAMS = {'base_lr': 0.0001,
                           'decay': 0.92,
                           'step': 1}
        
    AUG_SETTINGS = {
                    'gausian_blur': {'sigma': ((0.004, 0.4), (0.16, 1.6)),
                     'p': (0.05, 0.5)},
    
                    'motion_blur': {'p': (0, 0.4)},
                    
                    'color_jitter': {'brightness': ((0.98, 0.8), (1.01, 1.1)),
                                'contrast': ((0.97, 0.7), (1.017, 1.17)),
                                'saturation': ((0.94, 0.6),(1.04, 1.4)),
                                'hue': ((-0.004, -0.04), (0.004, 0.04)),
                                'p': (0.06, 0.6)},
                    
                    'thin_plate_spline': {'scale': (0.025, 0.25),
                                        'p': (0.035, 0.35)},
                    
                    'rotation': {'degrees': ((-1.7, -17), (1.7, 17)),
                                'p': (0.055, 0.55)},
                    
                    'perspective': {'distortion_scale': (0.025, 0.25),
                                    'p': (0.065, 0.65)},
                    
                    'cutout1' : {'scale': ((0, 0.02), (0, 0.07)),
                                'p': (0, 0.45)},
                    
                    'cutout2': {'scale': ((0, 0.015), (0, 0.06)),
                                'p': (0, 0.25)},
                                    
                    'resolution': {'min_res': 256,
                                   'max_res': 512},
                    'stages_num': 8,
                    'epochs_max': 16
                }
    
    
    
    
    
