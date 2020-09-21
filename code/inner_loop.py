import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.autograd import Variable

from layers import *
from score import *
from data_loading import *
from mlp import mlp

class InnerLoop(mlp):
	# This module performs the inner loop of MAML
	# The forward method updates weights with gradient steps on training data, 
	# then computes and returns a meta-gradient w.r.t. validation data

	def __init__(self, num_updates, step_size, feature_dim, layer, hidden):

		super(InnerLoop, self).__init__(feature_dim, layer, hidden)
		# Number of updates to be taken
		self.num_updates = num_updates

		# Step size for the updates
		self.step_size = step_size

		self.loss_fn = nn.MSELoss()

	def net_forward(self, x, weights=None):
		return super(InnerLoop, self).forward(x, weights)

	def forward_pass(self, in_, target, weights=None):
	
		# Run data through net, return loss and output
		input_var = torch.autograd.Variable(in_)
		target_var = torch.autograd.Variable(target)
		# Run the batch through the net, compute loss

		out, _ = self.net_forward(input_var, weights)
		loss = self.loss_fn(out, target_var)	
	
		return loss, out
	
	def forward(self, train_loader, val_loader):

		##### Test net before training, should be random accuracy ####
		#tr_pre_loss, tr_pre_acc = evaluate(self, train_loader)
		#val_pre_loss, val_pre_acc = evaluate(self, val_loader)
		
		fast_weights = OrderedDict((name, param) for (name, param) in self.named_parameters())
	
		for i in range(self.num_updates):
			
			#print 'inner step', i,
			in_, target = train_loader.__iter__().next()
				
			if i==0:
				loss, _ = self.forward_pass( in_, target )
				grads = torch.autograd.grad( loss, self.parameters(), create_graph=True )
			else:
				loss, _ = self.forward_pass(in_, target, fast_weights)
				grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)
			
			fast_weights = OrderedDict((name, param - self.step_size*grad) for ((name, param), grad) in zip(fast_weights.items(), grads))	
		##### Test net after training, should be better than random ####
		tr_post_loss, tr_post_acc = evaluate( self, train_loader, 0, fast_weights )
		val_post_loss, val_post_acc = evaluate( self, val_loader, 0, fast_weights ) 
		
		#print 'Train Inner Loss', tr_pre_loss, tr_post_loss, 'Train Inner Corr', tr_pre_acc, tr_post_acc
		#print 'Val Inner step Loss', val_pre_loss, val_post_loss, 'Val Inner step Acc', val_pre_acc, val_post_acc
		
		# Compute the meta gradient and return it
		in_, target = val_loader.__iter__().next()
		
		loss,_ = self.forward_pass( in_, target, fast_weights ) 
		loss = loss / in_.size()[0] # normalize loss
		
		grads = torch.autograd.grad( loss, self.parameters() )
		meta_grads = {name:g for ((name, _), g) in zip(self.named_parameters(), grads)}
		metrics = (tr_post_loss, tr_post_acc, val_post_loss, val_post_acc)
		#metrics = (0,0,0,0)

		return metrics, meta_grads

