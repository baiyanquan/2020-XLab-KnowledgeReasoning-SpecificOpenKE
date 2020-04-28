import openke
from openke.config import Trainer, Tester
from openke.module.model import TransE, TransR
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader
import numpy as np
import matplotlib.pyplot as plt

# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = "./benchmarks/OMKG/", 
	nbatches = 100,
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 1, 
	filter_flag = 1, 
	neg_ent = 25,
	neg_rel = 0)

# dataloader for test
test_dataloader = TestDataLoader(
	in_path = "./benchmarks/OMKG/",
	sampling_mode = 'link')

# define the model
transe = TransE(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 50,
	p_norm = 1, 
	norm_flag = True)

model_e = NegativeSampling(
	model = transe, 
	loss = MarginLoss(margin = 5.0),
	batch_size = train_dataloader.get_batch_size())

transr = TransR(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim_e = 50,
	dim_r = 100,
	p_norm = 1, 
	norm_flag = True,
	rand_init = False)

model_r = NegativeSampling(
	model = transr,
	loss = MarginLoss(margin = 3.0),
	batch_size = train_dataloader.get_batch_size()
)

# pretrain transe
# trainer = Trainer(model = model_e, data_loader = train_dataloader, train_times = 1000, alpha = 0.5, use_gpu = False)
trainer = Trainer(model = model_e, data_loader = train_dataloader, train_times = 1000, alpha = 1.0, use_gpu = False)
trainer.run()
parameters = transe.get_parameters()
transe.save_parameters("./result/transr_transe.json")

# train transr
# transr.set_parameters(parameters)
transr.ent_embeddings = transe.ent_embeddings
trainer = Trainer(model = model_r, data_loader = train_dataloader, train_times = 1000, alpha = 1.0, use_gpu = False)
trainer.run()
transr.save_checkpoint('./checkpoint/transr.ckpt')

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
np.savetxt('./embedding/TransR/entity_result.txt',entity_np)

relationship_np = np.array(relationship)
np.savetxt('./embedding/TransR/relationship_result.txt',relationship_np)

# test the model
transr.load_checkpoint('./checkpoint/transr.ckpt')
tester = Tester(model = transr, data_loader = test_dataloader, use_gpu = False)
tester.run_link_prediction(type_constrain = False)