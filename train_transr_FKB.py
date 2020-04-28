import openke
from openke.config import Trainer, Tester
from openke.module.model import TransE, TransR
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader
import numpy as np
import matplotlib.pyplot as plt
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
test_dataloader = TestDataLoader(
	in_path = "./benchmarks/FKB/",
	sampling_mode = 'link')

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
transr = TransR(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim_e = 30,
	dim_r = 50,
	p_norm = 1, 
	norm_flag = True,
	rand_init = False)

transr.load_rel_embeddings(rel_embedding)

model_r = NegativeSampling(
	model = transr,
	loss = MarginLoss(margin = 3.0),
	batch_size = train_dataloader.get_batch_size()
)

for k,v in model_r.named_parameters():
	if k=='model.rel_embeddings.weight':
		v.requires_grad=False

# train transr
# transr.set_parameters(parameters)
trainer = Trainer(model = model_r, data_loader = train_dataloader, train_times = 200, alpha = 1.0, use_gpu = False)
trainer.run()
transr.save_checkpoint('./checkpoint/fault_dataset_transr.ckpt')

epoch = trainer.epoch
loss = trainer.loss
line = plt.plot(epoch, loss, label=u'TransR')
plt.xlabel(u'epoch')
plt.ylabel(u'loss')
plt.show()
plt.savefig("./embedding/TransR/loss_epoch.png")

entity = trainer.model.model.ent_embeddings.weight.data
relationship = trainer.model.model.rel_embeddings.weight.data

entity_np = np.array(entity)
np.savetxt('./embedding/TransR/fault_dataset_entity_result.txt',entity_np)

relationship_np = np.array(relationship)
np.savetxt('./embedding/TransR/fault_dataset_relationship_result.txt',relationship_np)

# test the model
transr.load_checkpoint('./checkpoint/fault_dataset_transr.ckpt')
tester = Tester(model = transr, data_loader = test_dataloader, use_gpu = False)
tester.run_link_prediction(type_constrain = False)