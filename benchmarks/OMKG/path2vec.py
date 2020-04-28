import numpy as np
import torch
import torch.nn.functional as F

path = []
path2vec = []

entity2vec = np.loadtxt("../../embedding/TransR/entity_result.txt")

path_file = open("pathList.txt", "r")
tot = (int)(path_file.readline())
for i in range(tot):
    content = path_file.readline()
    content_list = content.strip().split()
    specific_path = []
    for j in content_list:
        specific_path.append(int(j))
    path.append(specific_path)

for i in range(len(path)):
    path_info = {}
    path_info['start'] = path[i][0]
    path_info['end'] = path[i][len(path[i]) - 1]
    vector = torch.tensor(entity2vec[path[i][0]])
    vector = F.normalize(vector, 2, 0)
    for j in range(1, len(path[i])):
        normalized_entity = torch.tensor(entity2vec[path[i][j]])
        normalized_entity = F.normalize(normalized_entity, 2, 0)
        vector += normalized_entity
    path_info['vector'] = F.normalize(vector, 2, 0)
    path2vec.append(path_info)

with open('path2vec.txt','w') as f:
    for i in range(len(path2vec)):
        f.write(str(path2vec[i]['start']) + '\t' + str(path2vec[i]['end']) + '\t')
        f.write(str(path2vec[i]['vector'].numpy().tolist()) + '\n')
f.close()
