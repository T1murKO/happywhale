# from sklearn.neighbors import NearestNeighbors
# import cv2
# import torch
# import json
# from tqdm import tqdm
# from transforms import get_eval_list

# input_size = (256, 256)

# def get_embedding(img_path):
#     img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
#     input = torch.unsqueeze(get_transform_list(img), 0).to(device)
#     embed = model(input).detach().cpu().numpy()

#     return embed



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