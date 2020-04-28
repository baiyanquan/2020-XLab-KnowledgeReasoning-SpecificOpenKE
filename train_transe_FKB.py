import openke
from openke.config import Trainer, Tester
from openke.module.model import TestTransE, TransE
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader
import numpy as np
import torch.nn as nn
import torch

# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = "./benchmarks/FKB/",
	nbatches = 100,
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 1, 
	filter_flag = 1, 
	neg_ent = 25,
	neg_rel = 0)

# dataloader for test
test_dataloader = TestDataLoader("./benchmarks/FKB/", "link")

path_vec_list = []
with open("./benchmarks/OMKG/path2vec.txt") as f:
	for line in f.readlines():
		path_vec_list.append(line.split('\t')[2].strip('[').strip(']\n').split(','))
f.close()

extract_path_vec_list = []
with open("./benchmarks/FKB/relation2id.txt") as f:
	f.readline()
	for line in f.readlines():
		extract_path_vec_list.append(path_vec_list[int(line.split('\t')[0])])
f.close()

rel_embedding = nn.Embedding.from_pretrained(torch.from_numpy(np.array(extract_path_vec_list).astype(dtype='float64')).float())


# define the model
transe = TransE(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 50,
	p_norm = 1, 
	norm_flag = True)

transe.load_rel_embeddings(rel_embedding)

# define the loss function
model = NegativeSampling(
	model = transe, 
	loss = MarginLoss(margin = 10.0),
	batch_size = train_dataloader.get_batch_size()
)

for k,v in model.named_parameters():
	if k=='model.rel_embeddings.weight':
		v.requires_grad=False

# train the model
trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 1000, alpha = 1.0, use_gpu = False)
trainer.run()

relationship = trainer.model.model.rel_embeddings.weight.data

entity = trainer.model.model.ent_embeddings.weight.data
relationship = trainer.model.model.rel_embeddings.weight.data

entity_np = np.array(entity)
np.savetxt('./embedding/TransE/fault_dataset_entity_result.txt',entity_np)

relationship_np = np.array(relationship)
np.savetxt('./embedding/TransE/fault_dataset_relationship_result.txt',relationship_np)

transe.save_checkpoint('./checkpoint/fault_dataset_transe.ckpt')

# test the model
transe.load_checkpoint('./checkpoint/fault_dataset_transe.ckpt')
tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = False)
tester.run_link_prediction(type_constrain = False)