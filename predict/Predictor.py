from .ModelLoader import ModelLoader
from .EntityAndRelationDataLoader import EntityAndRelationDataLoader
import numpy as np
import torch
import torch.nn.functional as F


class Predictor(object):
    def __init__(self):
        self.entity = []
        self.p_norm = 1

    def data_load(self, model_base_path='./embedding/', entity_relation_base_path = './benchmarks/FKB2/', o_m_base_path = './benchmarks/OMKG/'):
        model = ModelLoader(model_base_path)
        if model.load_model('TransR'):
            self.ent_embedding = model.get_ent_embedding()
            self.rel_embedding = model.get_rel_embedding()
            self.transfer_matrix = model.get_transfer_matrix()
        else:
            print('Read embeddings error')
        self.entityAndRelation = EntityAndRelationDataLoader(entity_relation_base_path, o_m_base_path)
        self.entity = self.entityAndRelation.acquire_entity()
        self.path = self.entityAndRelation.acquire_path()
        self.o_m_entity = self.entityAndRelation.acquire_o_m_base_path()

    def predict(self, target_performance):
        path_detail = []
        with open('./benchmarks/FKB2/pathList.txt') as f:
            f.readline()
            i = 0
            for line in f.readlines():
                path_detail.append(line.rstrip().split('\t'))
        f.close()

        fault_list = ['cpu', 'mem', 'network', 'disk', 'k8s']

        prefix = 'http://10.60.38.181/service/'
        temp = target_performance.split(':')[0].split('-')
        performance_service = temp[0]
        for i in range(1, len(temp) - 1):
            performance_service += '-' + temp[i]
        path_list = self.path[str(self.o_m_entity.index(prefix + performance_service))]
        h = self.ent_embedding(torch.tensor(np.full(len(self.entity), self.entity.index(target_performance))))
        h = h.view(-1, 1, 30)
        t = self.ent_embedding(torch.tensor(np.arange(self.ent_embedding.weight.shape[0])))
        t = t.view(-1, 1, 30)
        result = []
        for i in path_list:
            r = self.rel_embedding(torch.tensor(np.full(len(self.entity), i)))
            r = F.normalize(r, 2, -1)
            r_transfer = self.transfer_matrix(torch.tensor(np.full(len(self.entity), i)))
            r_transfer = r_transfer.view(-1,30,50)
            h_transfer = torch.matmul(h, r_transfer).view(-1, 50)
            t_transfer = torch.matmul(t, r_transfer).view(-1, 50)
            h_transfer = F.normalize(h_transfer, 2, -1)
            t_transfer = F.normalize(t_transfer, 2, -1)
            score = (h_transfer + r) - t_transfer
            score = torch.norm(score, 1, -1).flatten()
            path_str = ""
            for j in path_detail[i]:
                path_str += self.o_m_entity[int(j)] + '\t'
            path_str = path_str.rstrip()
            score_np = np.array(score)

            index_list = [-1, 0, 0, 0]
            index_value = [0, score_np.max(), score_np.max(), score_np.max()]
            for k in range(1, len(index_list)):
                for l in range(len(score_np)):
                    if l != index_list[k-1] and score_np[l] > index_value[k-1] and score_np[l] < index_value[k]:
                        index_list[k] = l
                        index_value[k] = score_np[l]
            for k in range(1, len(index_list)):
                if self.entity[index_list[k]] in fault_list:
                    result.append({"path": path_str, "fault": self.entity[index_list[k]], "score": index_value[k]})
        return result
