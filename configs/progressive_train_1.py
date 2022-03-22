class config:
    SEED = 42
    
    DATA_PATH = '/root/kaggle/happywhale/train_1.beton'
    SAVE_PATH = '/root/kaggle/happywhale/effnetv2_m'
    
    EMBED_DIM = 512
    CLASS_NUM = 15587
    EPOCHES_NUM = 20
    BATCH_SIZE = 92
    
    IS_RESUME = True
    LOAD_PATH = '/root/kaggle/happywhale/effnetv2_m/epoch_16_model.pt'
    START_EPOCH = 16
   
    
    BACKBONE_NAME = 'effnetv2_m'
    BACKBONE_PARAMS = {'pretrained': True}
    
    TRIPLET_MARGIN = 0.225
    
    HEAD_NAME = 'adacos'
    HEAD_PARAMS = {'fixed_scale': False,
                    'feat_dim': EMBED_DIM,
                    'num_classes': CLASS_NUM}
    
    POOLING_NAME = 'gem'
    POOLING_PARAMS = {}
    
    
    LR_SCHEDULER_NAME = 'ramp'
    LR_SCHEDULER_PARAMS = {'lr_start': 0.000001,
                        'lr_max': 0.0001,
                        'lr_min': 0.00001,
                        'lr_ramp_ep': 5,
                        'lr_sus_ep': 5,
                        'lr_decay': 0.925}
        
    AUG_SETTINGS = {
                    'gausian_blur': {'sigma': ((0, 0.1), (0, 1)),
                                    'p': (0, 0.3)},
                    
                    'motion_blur': {'p': (0, 0)},
                    
                    'color_jitter': {'brightness': ((1, 0.97), (1, 1.03)),
                                'contrast': ((1, 0.9), (1, 1.05)),
                                'saturation': ((1, 0.85), (1, 1.15)),
                                'hue': ((0, -0.01), (0, 0.01)),
                                'p': (0, 0.35)},
                    
                    'thin_plate_spline': {'scale': (0.012, 0.12),
                                        'p': (0, 0.1)},
                    
                    'rotation': {'degrees': ((0, -7), (0, 7)),
                                'p': (0, 0.2)},
                    
                    'perspective': {'distortion_scale': (0, 0.15),
                                    'p': (0, 0.25)},
                    
                    'cutout1' : {'scale': ((0, 0.015), (0, 0.06)),
                                'p': (0, 0.25)},
                    
                    'cutout2': {'scale': ((0, 0.015), (0, 0.06)),
                                'p': (0, 0)},
                    
                    'resolution': {'min_res': 192,
                                   'max_res': 256},
                    'stages_num': 8,
                    'epochs_max': 16
                }
    
    

    
    
    
    
