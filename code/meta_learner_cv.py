import time
import argparse
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
import os
import glob
from torch.autograd import Variable
import sys
import torch.nn as nn
import pickle
import copy
from data_loading import *
from utils import *
from score import *
from inner_loop import InnerLoop
from mlp import mlp

# This is the meta-learner class

class dataset(object):
	def __init__(self, feature, label):
		super(self.__class__, self).__init__()
		self.feature = feature
		self.label = label

class MetaLearner(object):
	def __init__(self, meta_dataset, fs_dataset, K, meta_lr, inner_lr, layer, hidden, tissue_num, meta_batch_size, inner_batch_size, num_updates, num_inner_updates, tissue_index_list, patience=3, num_trials=10 ):

		super(self.__class__, self).__init__()
		
		self.meta_dataset = meta_dataset
		self.fs_dataset = fs_dataset

		self.meta_batch_size = meta_batch_size
		self.inner_batch_size = inner_batch_size
		self.num_updates = num_updates
		self.num_inner_updates = num_inner_updates
		self.num_trials = num_trials
		self.hidden = hidden
		self.patience = patience
		self.feature_num = self.fs_dataset.feature.shape[1]

		self.K = K
		self.meta_lr = meta_lr
		self.inner_lr = inner_lr
		self.layer = layer
		self.hidden = hidden
		self.tissue_index_list = tissue_index_list
		self.tissue_num = tissue_num

		self.observed_tissue_model = mlp(self.feature_num, layer, hidden)
		self.observed_opt = torch.optim.Adam(self.observed_tissue_model.parameters(), lr=self.meta_lr, betas=(0.9, 0.99), eps=1e-05)
		self.inner_net = InnerLoop(self.num_inner_updates, self.inner_lr, self.feature_num, layer, hidden)

		#torch.cuda.manual_seed(args.seed)
		self.observed_tissue_model.cuda()
		self.inner_net.cuda()

	def zero_shot_test(self, unseen_train_loader, unseen_vali_loader, unseen_test_loader):
		
		unseen_tissue_model = mlp( self.feature_num, self.layer, self.hidden )

		# First need to copy the original meta learning model
		unseen_tissue_model.copy_weights( self.observed_tissue_model )
		unseen_tissue_model.cuda()
		unseen_tissue_model.eval()

		train_performance = evaluate_cv( unseen_tissue_model, unseen_train_loader)
		vali_performance = evaluate_cv( unseen_tissue_model, unseen_vali_loader)
		test_performance = evaluate_cv( unseen_tissue_model, unseen_test_loader)

		return train_performance, vali_performance, test_performance, np.mean(tissue_loss)	

	def meta_update(self, test_loader, ls):

		#Walk in 'Meta update' function
		in_, target = test_loader.__iter__().next()
	
		# We use a dummy forward / backward pass to get the correct grads into self.net
		loss, out = forward_pass(self.observed_tissue_model, in_, target)
	
		# Unpack the list of grad dicts
		gradients = { k: sum(d[k] for d in ls) for k in ls[0].keys() }
	
		#for k, val, in gradients.items():
		#	gradients[k] = val / args.meta_batch_size
		#	print k,':',gradients[k]

		# Register a hook on each parameter in the net that replaces the current dummy grad
		# with our grads accumulated across the meta-batch
		hooks = []
	
		for (k,v) in self.observed_tissue_model.named_parameters():
		
			def get_closure():
				key = k
				def replace_grad(grad):
					return gradients[key]
				return replace_grad
		
			hooks.append(v.register_hook(get_closure()))
	
		# Compute grads for current step, replace with summed gradients as defined by hook
		self.observed_opt.zero_grad()
	
		loss.backward()
		# Update the net parameters with the accumulated gradient according to optimizer
		self.observed_opt.step()
		# Remove the hooks before next training phase
		for h in hooks:
			h.remove()

	def unseen_tissue_learn(self, unseen_train_loader, unseen_test_loader):

		unseen_tissue_model = mlp( self.feature_num, self.layer, self.hidden )

		# First need to copy the original meta learning model
		unseen_tissue_model.copy_weights( self.observed_tissue_model )
		unseen_tissue_model.cuda()
		#unseen_tissue_model.train()
		unseen_tissue_model.eval()

		unseen_opt = torch.optim.SGD(unseen_tissue_model.parameters(), lr=self.inner_lr)
		#unseen_opt = torch.optim.Adam(unseen_tissue_model.parameters(), lr=self.inner_lr, betas=(0.9, 0.99), eps=1e-05)

		# Here test_feature and test_label contains only one tissue info
		#unseen_train_loader, unseen_test_loader = get_unseen_data_loader(test_feature, test_label, K, args.inner_batch_size)
		for i in range(self.num_inner_updates):

			in_, target = unseen_train_loader.__iter__().next()
			loss, _  = forward_pass( unseen_tissue_model, in_, target )
			unseen_opt.zero_grad()
			loss.backward()
			unseen_opt.step()

		# Test on the rest of cell lines in this tissue (unseen_test_loader)
		mtrain_loss, mtrain_pear_corr, mtrain_spearman_corr, _, _ = evaluate_new( unseen_tissue_model, unseen_train_loader,1 )
		mtest_loss, mtest_pear_corr, mtest_spearman_corr, test_prediction, test_true_label = evaluate_new( unseen_tissue_model, unseen_test_loader,0 )

		return mtrain_loss, mtrain_pear_corr, mtrain_spearman_corr, mtest_loss, mtest_pear_corr, mtest_spearman_corr, test_prediction, test_true_label
	
	def train(self):

		best_train_loss_test_corr, best_train_corr_test_corr = 0, 0
		best_train_corr_test_scorr, best_train_scorr_test_scorr = 0, 0
	
		best_train_corr_model = ''
	
		unseen_train_loader, unseen_test_loader = get_unseen_data_loader( self.fs_dataset.feature, self.fs_dataset.label, self.K )
		#unseen_train_loader, unseen_test_loader = get_unseen_data_loader( self.fs_dataset.feature, self.fs_dataset.label, self.fs_catdata, self.K )

		# Here the training process starts
		best_fewshot_train_corr, best_fewshot_train_loss = -2, 1000
		best_fewshot_test_corr, best_fewshot_test_loss = -2, 1000 
		best_train_loss_epoch, best_train_corr_epoch = 0, 0

		best_fewshot_train_spearman_corr, best_fewshot_test_spearman_corr = -2, -2
		best_train_spearman_corr_epoch = 0

		train_loss, train_corr = np.zeros((self.num_updates,)), np.zeros((self.num_updates,))
		test_loss, test_corr = np.zeros((self.num_updates,)), np.zeros((self.num_updates,))
		train_spearman_corr, test_spearman_corr = np.zeros((self.num_updates,)), np.zeros((self.num_updates,))

		for epoch in range( self.num_updates ):

			# Collect a meta batch update
			grads = []
			meta_train_loss, meta_train_corr, meta_val_loss, meta_val_corr = np.zeros((self.meta_batch_size,)), np.zeros((self.meta_batch_size,)), np.zeros((self.meta_batch_size,)), np.zeros((self.meta_batch_size,))

			self.inner_net.copy_weights( self.observed_tissue_model )
			for i in range( self.meta_batch_size ):

				observed_train_loader, observed_test_loader = get_observed_data_loader( self.meta_dataset.feature, self.meta_dataset.label, self.tissue_index_list, self.K, self.inner_batch_size, self.tissue_num )

				#self.inner_net.copy_weights( self.observed_tissue_model )

				metrics, g = self.inner_net.forward( observed_train_loader, observed_test_loader )
				grads.append( g )

				meta_train_loss[i], meta_train_corr[i], meta_val_loss[i], meta_val_corr[i] = metrics

			# Perform the meta update		
			self.meta_update( observed_test_loader, grads )

			#meta_train_loss_mean, meta_train_corr_mean, meta_val_loss_mean, meta_val_corr_mean = meta_train_loss.mean(), meta_train_corr.mean(), meta_val_loss.mean(), meta_val_corr.mean()

			## Evaluate K shot test tasks
			train_loss[epoch], train_corr[epoch], train_spearman_corr[epoch], test_loss[epoch], test_corr[epoch], test_spearman_corr[epoch], _, _ = self.unseen_tissue_learn( unseen_train_loader, unseen_test_loader )

			if test_loss[epoch] < best_fewshot_test_loss:
				best_fewshot_test_loss = test_loss[epoch]
	   
			if test_corr[epoch] > best_fewshot_test_corr:
				best_fewshot_test_corr = test_corr[epoch]
 
			if train_loss[epoch] < best_fewshot_train_loss:
				best_fewshot_train_loss = train_loss[epoch]
				best_train_loss_epoch = epoch
	
			if train_corr[epoch] > best_fewshot_train_corr:
	 			best_fewshot_train_corr = train_corr[epoch]
				best_train_corr_epoch = epoch
				best_train_corr_model = self.observed_tissue_model 
		
			if train_spearman_corr[epoch] > best_fewshot_train_spearman_corr:
				best_fewshot_train_spearman_corr = train_spearman_corr[epoch]
				best_train_spearman_corr_epoch = epoch

			if test_spearman_corr[epoch] > best_fewshot_test_spearman_corr:
				best_fewshot_test_spearman_corr = test_spearman_corr[epoch]	
				best_test_spearman_epoch = epoch	

			print 'Few shot', epoch, 'train_loss:', float('%.3f'%train_loss[epoch]), 'train_pearson:', float('%.3f'%train_corr[epoch]), 'train_spearman:', float('%.3f'%train_spearman_corr[epoch]),
			print 'test_loss:', float('%.3f'%test_loss[epoch]), 'test_pearson:', float('%.3f'%test_corr[epoch]), 'test_spearman:', float('%.3f'%test_spearman_corr[epoch])

		best_train_loss_test_corr = test_corr[ best_train_loss_epoch ]
		best_train_corr_test_corr = test_corr[ best_train_corr_epoch ]
		best_train_corr_test_scorr = test_spearman_corr[ best_train_corr_epoch ]
		best_train_scorr_test_scorr = test_spearman_corr[ best_train_spearman_corr_epoch ]
		
		print '--trial summerize--', 'best_train_loss_test_corr:', float('%.3f'%best_train_loss_test_corr), 'best_train_corr_test_corr', float('%.3f'%best_train_corr_test_corr), 'best_train_corr_test_scorr', float('%.3f'%best_train_corr_test_scorr), 'best_train_scorr_test_corr', float('%.3f'%best_train_scorr_test_scorr)
		
		return best_train_loss_test_corr, best_train_corr_test_corr, best_train_corr_test_scorr, best_train_scorr_test_scorr, best_train_corr_model
