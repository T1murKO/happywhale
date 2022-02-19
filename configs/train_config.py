# EffNetB0	224
# EffNetB1	240
# EffNetB2	260
# EffNetB3	300
# EffNetB4	380
# EffNetB5	456
# EffNetB6	528
# EffNetB7	600

class config:
    SEED = 42
    
    CSV_PATH = '/root/happywhale/data/train.csv'
    IAMGES_PATH = '/root/cropped_train_images'
    SAVE_PATH = '/root/effnetb6'
    MASK_PASS = None
    CROP_PASS = None
    
    EMBED_DIM = 512
    INPUT_SIZE = (512, 512)
    CLASS_NUM = 15587
    EPOCHES_NUM = 20
    BATCH_SIZE = 36
    
    IS_RESUME = False
    LOAD_PATH = None
    START_EPOCH = 0

    MIXED_PRESICION = True
    
    
    BACKBONE_NAME = 'effnetv1_b6'
    BACKBONE_PARAMS = {'pretrained': True}
    
    HEAD_NAME = 'adacos'
    HEAD_PARAMS = {'fixed_scale': False,
                    'feat_dim': EMBED_DIM,
                    'num_classes': CLASS_NUM}
    
    POOLING_NAME = 'gem'
    POOLING_PARAMS = {}
    
    SCHEDULER_NAME = 'ramp'
    SCHEDULER_PARAMS = {}
    
