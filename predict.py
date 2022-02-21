from sklearn.neighbors import NearestNeighbors
import cv2
import torch
import json
from tqdm import tqdm
from utils.dataset import TestImageDataset, DummyDataset, DummyDataset2
from torch.utils.data import DataLoader
from utils.transforms import get_infer_list
from utils import TrainImageDataset
from modules.zoo import *
from modules import Model
from configs.infer_config import config
from configs.train_config import config as model_config
import pandas as pd
import os
from os.path import join
import numpy as np
from modules.zoo import get_backbone,\
                        get_pooling, \
                        get_head, \
                        get_scheduler

from modules import Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device.type)
if device.type == 'cuda':
    NUM_GPU = torch.cuda.device_count()
    print(f'[INFO] number of GPUs found: {NUM_GPU}')
    if NUM_GPU > 1:
        DISTRIBUTED = True
        config.BATCH_SIZE = config.BATCH_SIZE * NUM_GPU
    else:
        DISTRIBUTED = False


if not os.path.exists(config.SAVE_PATH):
    os.mkdir(config.SAVE_PATH)

with open(os.path.join(config.SAVE_PATH + '/predict_config.json'), 'w') as f:
    predict_config = {x:dict(config.__dict__)[x] for x in dict(config.__dict__) if not x.startswith('_')}
    json.dump(predict_config, f)

with open(os.path.join(config.SAVE_PATH + '/predict_config.json'), 'r') as f:
    predict_config = json.load(f)

print('[INFO] Prediction config', json.dumps(predict_config, indent=3, sort_keys=True))

if not config.NEIGHBORS_PATH:

    data_csv = pd.read_csv(config.CSV_PATH)

    backbone, backbone_dim = get_backbone(model_config.BACKBONE_NAME, model_config.BACKBONE_PARAMS)
    pooling = get_pooling(model_config.POOLING_NAME, model_config.POOLING_PARAMS)
    head = get_head(model_config.HEAD_NAME, model_config.HEAD_PARAMS)
    model = Model(model_config.CLASS_NUM, backbone, pooling, head, embed_dim=model_config.EMBED_DIM, backbone_dim=backbone_dim)
    model.load_state_dict(torch.load(config.MODEL_PATH).module.state_dict())
    model.to(device)
    if DISTRIBUTED:
        model = torch.nn.DataParallel(model)
        
    model.eval()

    transforms = get_infer_list(input_size=config.INPUT_SIZE)

    train_dataset = TrainImageDataset(data_csv,
                                config.TRAIN_IMAGES_PATH,
                                transform=transforms)
    
    test_dataset = TestImageDataset(config.TEST_IMAGES_PATH,
                                transform=transforms)

    # train_dataset = DummyDataset(config.INPUT_SIZE, num_samples=100)

    # test_dataset = DummyDataset2(config.INPUT_SIZE, num_samples=50)

    train_loader = DataLoader(train_dataset,
                            batch_size=config.BATCH_SIZE,
                            shuffle=False,
                            num_workers=os.cpu_count(),
                            pin_memory=False)

    test_loader = DataLoader(test_dataset,
                            batch_size=config.BATCH_SIZE,
                            shuffle=False,
                            num_workers=os.cpu_count(),
                            pin_memory=False)

    target_to_id = json.loads(open('./data/json/target_to_id.json', 'r').read())
    target_to_id = {int(key): target_to_id[key] for key in target_to_id}


    print('[INFO] Processing embeddings for train images')

    train_targets = []
    train_embeddings = []

    for images, targets in tqdm(train_loader):
        images = images.to(device)
        embeddings = model(images).detach().cpu().numpy()
        
        train_embeddings.append(embeddings)
        train_targets.append(targets)

    train_embeddings = np.squeeze(np.concatenate(train_embeddings))
    train_targets = np.concatenate(train_targets)

    neigh = NearestNeighbors(n_neighbors=config.KNN_NUM,metric='cosine')
    neigh.fit(train_embeddings)
        


    print('[INFO] Processing test images')


    test_ids = []
    test_nn_distances = []
    test_nn_idxs = []

    for images, img_names in tqdm(test_loader):
        images = images.to(device)
        embeddings = model(images).detach().cpu().numpy()
        distances, idxs = neigh.kneighbors(embeddings, config.KNN_NUM, return_distance=True)
        test_ids.append(img_names)
        test_nn_idxs.append(idxs)
        test_nn_distances.append(distances)


    test_nn_distances = np.concatenate(test_nn_distances)
    test_nn_idxs = np.squeeze(np.concatenate(test_nn_idxs))
    test_ids = np.squeeze(np.concatenate(test_ids))

    print('[INFO] Building nearest neighbors dataframe')
    
    test_df = []
    for i in tqdm(range(len(test_ids))):
        id_ = test_ids[i]
        targets = train_targets[test_nn_idxs[i]]
        distances = test_nn_distances[i]
        subset_preds = pd.DataFrame(np.stack([targets,distances],axis=1),columns=['target','distances'])
        subset_preds['image'] = id_
        test_df.append(subset_preds)
        
    test_df = pd.concat(test_df).reset_index(drop=True)
    test_df['confidence'] = 1-test_df['distances']
    test_df = test_df.groupby(['image','target']).confidence.max().reset_index()
    test_df = test_df.sort_values('confidence',ascending=False).reset_index(drop=True)
    test_df['target'] = test_df['target'].map(target_to_id)
    test_df.to_csv(join(config.SAVE_PATH, 'test_neighbors.csv'))

else:
    print('[INFO] Neighbors dataframe specified, loading...')
    test_df = pd.read_csv(config.NEIGHBORS_PATH)
    

print('[INFO] Processing nearest neighbors dataframe')

sample_list = ['938b7e931166', '5bf17305f073', '7593d2aee842', '7362d7a01d00','956562ff2888']
exclude = ['37c7aba965a5', '114207cab555']

n_new = 0
predictions = {}
for i,row in tqdm(test_df.iterrows()):
    if row.target in exclude:
        continue
    if row.image in predictions:
        if len(predictions[row.image])==5:
            continue
        predictions[row.image].append(row.target)
    elif row.confidence > config.THRESHOLD:
        predictions[row.image] = [row.target,'new_individual']
    else:
        n_new += 1
        predictions[row.image] = ['new_individual',row.target]
        
for x in tqdm(predictions):
    if len(predictions[x])<5:
        remaining = [y for y in sample_list if y not in predictions]
        predictions[x] = predictions[x]+remaining
        predictions[x] = predictions[x][:5]
    predictions[x] = ' '.join(predictions[x])

print(f'[INFO] New individual share {round(n_new / len(predictions), 3)}')

predictions = pd.Series(predictions).reset_index()
predictions.columns = ['image','predictions']
predictions.to_csv(join(config.SAVE_PATH, 'submission.csv'), index=False)


