import numpy as np
import os
import torch
import torch.nn as nn

class ModelLoader(object):
    def __init__(self, base_path):
        self.base_path = base_path

    def load_model(self, model_name='TransR'):
        if os.path.exists(self.base_path + model_name + '/fault_dataset_entity_result.txt') and os.path.exists(self.base_path + model_name + '/fault_dataset_relationship_result.txt'):
            self.ent_embedding = nn.Embedding.from_pretrained(torch.from_numpy(np.loadtxt(self.base_path + model_name + '/fault_dataset_entity_result.txt').astype(dtype='float64')).float())
            self.rel_embedding = nn.Embedding.from_pretrained(torch.from_numpy(np.loadtxt(self.base_path + model_name + '/fault_dataset_relationship_result.txt').astype(dtype='float64')).float())
            self.transfer_matrix = nn.Embedding.from_pretrained(torch.from_numpy(np.loadtxt(self.base_path + model_name + '/fault_dataset_transfer_matrix_result.txt').astype(dtype='float64')).float())
            return True
        else:
            return False

    def get_ent_embedding(self):
        return self.ent_embedding

    def get_rel_embedding(self):
        return self.rel_embedding

    def get_transfer_matrix(self):
        return self.transfer_matrix
