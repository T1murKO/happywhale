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
    SAVE_PATH = '/content/drive/MyDrive/senet154/'
    MASK_PASS = None
    CROP_PASS = None
    
    EMBED_DIM = 512
    INPUT_DIM = (512, 512)
    CLASS_NUM = 15587
    EPOCHES_NUM = 20
    BATCH_SIZE = 16
    
    IS_RESUME = False
    LOAD_PATH = None
    START_EPOCH = 0
    
    
    BACKBONE_NAME = 'effnetv1_b6'
    BACKBONE_PARAMS = {'pretrained': True}
    
    HEAD_NAME = 'adacos'
    HEAD_PARAMS = {'fixed': False}
    
    POOLING_NAME = 'gem'
    
    SCHEDULER_NAME = 'ramp'
    SCHEDULER_PARAMS = {'batch_size': BATCH_SIZE}
    
    
    