class config:
    SEED = 42
    
    DATA_PATH = '/root/kaggle/happywhale/data/seg_train.beton'
    SAVE_PATH = '/root/kaggle/senet154'
    
    EMBED_DIM = 512
    INPUT_SIZE = (512, 512)
    CLASS_NUM = 15587
    EPOCHES_NUM = 20
    BATCH_SIZE = 30
    
    IS_RESUME = False
    LOAD_PATH = None
    START_EPOCH = 0

    MIXED_PRESICION = True
    
    
    BACKBONE_NAME = 'senet154'
    BACKBONE_PARAMS = {'pretrained': 'imagenet'}
    
    HEAD_NAME = 'adacos'
    HEAD_PARAMS = {'fixed_scale': False,
                    'feat_dim': EMBED_DIM,
                    'num_classes': CLASS_NUM}
    
    POOLING_NAME = 'gem'
    POOLING_PARAMS = {}
    
    SCHEDULER_NAME = 'ramp'
    SCHEDULER_PARAMS = {}
