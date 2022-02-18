# EffNetB0	224
# EffNetB1	240
# EffNetB2	260
# EffNetB3	300
# EffNetB4	380
# EffNetB5	456
# EffNetB6	528
# EffNetB7	600

class Config:
    SEED = 42
    
    CSV_PATH = '/content/happywhale/data/train.csv'
    IAMGES_PATH = '/content/train_images-256-256'
    SAVE_PATH = '/content/drive/MyDrive/seresnet50_baseline'
    MASK_PASS = None
    CROP_PASS = None
    
    EMBED_DIM = 512
    INPUT_DIM = (224, 224)
    CLASS_NUM = 15587
    EPOCHES_NUM = 20
    BATCH_SIZE = 16
    
    IS_RESUME = False
    LOAD_PATH = None
    START_EPOCH = 0

    MIXED_PRESICION = False
    
    
    BACKBONE_NAME = 'se_resnext50'
    BACKBONE_PARAMS = {'pretrained': None,
                        'inchannels': 4}
    
    HEAD_NAME = 'adacos'
    HEAD_PARAMS = {'fixed_scale': False,
                    'feat_dim': EMBED_DIM,
                    'num_classes': CLASS_NUM}
    
    POOLING_NAME = 'gem'
    POOLING_PARAMS = {}
    
    SCHEDULER_NAME = 'ramp'
    SCHEDULER_PARAMS = {'batch_size': BATCH_SIZE}
    
    

config = Config()
print(dict(Config.__dict__)