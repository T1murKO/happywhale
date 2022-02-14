import imgaug.augmenters as iaa
from torchvision import transforms as T


def get_augmentation_list(input_size=(256, 256)):
    transform_list = T.Compose([             
        iaa.Sequential([
            iaa.Sequential([
            iaa.Sometimes(0.3, iaa.AverageBlur(k=(3,3))),
            iaa.Sometimes(0.3, iaa.MotionBlur(k=(3,5))),
            iaa.Add((-10, 10), per_channel=0.5),
            iaa.Multiply((0.9, 1.1), per_channel=0.5),
            iaa.Sometimes(0.2, iaa.Affine(
                scale={'x': (0.9,1.1), 'y': (0.9,1.1)},
                translate_percent={'x': (-0.05,0.05), 'y': (-0.05,0.05)},
                shear=(-10,10),
                rotate=(-10,10)
                )),
            iaa.Sometimes(0.2, iaa.Grayscale(alpha=(0.8,1.0))),
            ], random_order=True),
            iaa.size.Resize(input_size, interpolation='cubic')
        ]).augment_image,     
        T.ToTensor()
    ])
    
    return transform_list


def get_eval_list(input_size=(256, 256)):
    transforms_list_eval = T.Compose([             
    iaa.Sequential([
        iaa.size.Resize(input_size, interpolation='cubic')
    ]).augment_image,     
    T.ToTensor()
    ])
    
    return transforms_list_eval