from sklearn.neighbors import NearestNeighbors
import cv2
import torch
import json
from tqdm import tqdm
from utils.transforms import get_infer_list
from modules.zoo import *
from modules import Model
from configs.infer_config import config
import pandas as pd


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
if device == 'cuda':
    NUM_GPU = torch.cuda.device_count()
    
    if NUM_GPU > 1:
        DISTRIBUTED = True
        config.BATCH_SIZE = config.BATCH_SIZE * NUM_GPU
    else:
        DISTRIBUTED = False


data_csv = pd.read_csv(config.CSV_PATH)
model = torch.load(config.MODEL_PATH).to(device)


# train_folder = '/content/train_images-256-256/'

# img_to_target = json.loads(open('/content/happywhale/data/img_to_target.json', 'r').read())

# train_targets = []
# train_embeddings = []
# for filename in tqdm(os.listdir(train_folder)):
#     embeddings = get_embedding(join(train_folder, filename))
#     targets = img_to_target[filename]
#     train_embeddings.append(embeddings)
#     train_targets.append(targets)

# train_embeddings = np.array(train_embeddings)
# train_targets = np.array(train_targets)

# neigh = NearestNeighbors(n_neighbors=75,metric='cosine')
# neigh.fit(train_embeddings)


# test_folder = '/content/test_images-256-256'
# img_to_target = json.loads(open('/content/happywhale/data/img_to_id.json', 'r').read())

# test_ids = []
# test_nn_distances = []
# test_nn_idxs = []
# for filename in tqdm(os.listdir(test_folder)):
#     embedding = get_embedding(join(test_folder, filename))
#     id = filename
#     embedding = embedding
#     distance,idx = neigh.kneighbors(embedding, 75, return_distance=True)
#     test_ids.append(id)
#     test_nn_idxs.append(idx)
#     test_nn_distances.append(distance)

# test_nn_distances = np.array(test_nn_distances)
# test_nn_idxs = np.array(test_nn_idxs)
# test_ids = np.array(test_ids)