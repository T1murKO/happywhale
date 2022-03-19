class config:
    SEED = 42
    
    DATA_PATH = '/root/kaggle/happywhale/data/seg_train.beton'
    SAVE_PATH = '/root/kaggle/effnetv2_m'
    
    EMBED_DIM = 512
    INPUT_SIZE = (512, 512)
    CLASS_NUM = 15587
    EPOCHES_NUM = 20
    BATCH_SIZE = 30
    
    IS_RESUME = False
    LOAD_PATH = None
    START_EPOCH = 0

    MIXED_PRESICION = True
    
    
    BACKBONE_NAME = 'effnetv2_m'
    BACKBONE_PARAMS = {'pretrained': True}
    
    TRIPLET_MARGIN = 0.3
    
    HEAD_NAME = 'adacos'
    HEAD_PARAMS = {'fixed_scale': False,
                    'feat_dim': EMBED_DIM,
                    'num_classes': CLASS_NUM}
    
    POOLING_NAME = 'gem'
    POOLING_PARAMS = {}
    
    SCHEDULER_NAME = 'ramp'
    LR_SCHEDULER_PARAMS = {'lr_start': 0.000001,
                        'lr_max': 0.0001,
                        'lr_min': 0.00001,
                        'lr_ramp_ep': 5,
                        'lr_sus_ep': 5,
                        'lr_decay': 0.925}
    
    RESOLUTION_SCHEDULER_PARAMS = {
                                'min_res': 128,
                                'max_res': 256,
                                'start_ramp': 3,
                                'end_ramp': 12}
    
    # RESOLUTION_SCHEDULER_PARAMS = {
    #                         'min_res': 256,
    #                         'max_res': 512,
    #                         'start_ramp': 4,
    #                         'end_ramp': 14}
    
    
    
    
