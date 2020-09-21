import sys
import os
import numpy as np
import torch
import torch.utils.data as du
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from layers import *

class mlp(nn.Module):

	def __init__(self, feature_dim, layer, hidden):
		
		super(mlp, self).__init__()

		self.add_module( 'linear1', nn.Linear(feature_dim, hidden) )
		#self.add_module( 'bn1', nn.BatchNorm1d(hidden) )

		if layer == 2:
			self.add_module( 'linear2', nn.Linear(hidden, hidden) )
			#self.add_module( 'bn2', nn.BatchNorm1d(hidden) )
	
		self.add_module( 'linear3', nn.Linear(hidden, 1) )

		self.layer = layer
		self.loss_fn = nn.MSELoss()

		self._init_weights() 

	def forward(self, x, weights = None):

		if weights == None:
			
			hidden = F.relu( self._modules['linear1'](x) )
			#out = F.tanh( self._modules['bn1'](out) )
			#hidden = F.relu( self._modules['bn1']( self._modules['linear1'](x) ) )
			
			if self.layer == 2:
				hidden = F.relu( self._modules['linear2']( hidden ) )
				#out = F.tanh( self._modules['bn2'](out) )
				#hidden = F.relu( self._modules['bn2']( self._modules['linear2']( hidden ) ) )

			#out = self._modules['linear3']( hidden )	
			#out = F.tanh( out )
			out = self._modules['linear3']( hidden )

		else:
			
			#hidden = F.tanh( linear(x, weights['linear1.weight'], weights['linear1.bias']) )
			#out = batchnorm(out, weight = weights['bn1.weight'], bias = weights['bn1.bias'], momentum=1)
			#hidden = linear(x, weights['linear1.weight'], weights['linear1.bias'])
			#hidden = relu( batchnorm( hidden, weight = weights['bn1.weight'], bias = weights['bn1.bias']) )
			hidden = relu( linear(x, weights['linear1.weight'], weights['linear1.bias']) )

			if self.layer == 2:
				#hidden = F.tanh( linear(hidden, weights['linear2.weight'], weights['linear2.bias']) )
				#out = batchnorm(out, weight = weights['bn2.weight'], bias = weights['bn2.bias'], momentum=1)
				#hidden = linear( hidden, weights['linear2.weight'], weights['linear2.bias'] )
				#hidden = relu( batchnorm( hidden, weight = weights['bn2.weight'], bias = weights['bn2.bias']) )
				hidden = relu( linear( hidden, weights['linear2.weight'], weights['linear2.bias'] ) )

			#out = F.tanh( linear(hidden, weights['linear3.weight'], weights['linear3.bias']) )
			out = linear(hidden, weights['linear3.weight'], weights['linear3.bias'])

		return out, hidden

	def copy_weights(self, net):
		# Set this module's weights to be the same as those of 'net'
		for m_from, m_to in zip(net.modules(), self.modules()):
			if isinstance(m_to, nn.Linear) or isinstance(m_to, nn.BatchNorm1d):

				m_to.weight.data = m_from.weight.data.clone()

				if m_to.bias is not None:
					m_to.bias.data = m_from.bias.data.clone()

	def net_forward(self, x, weights=None):
		return self.forward(x, weights)

	def _init_weights(self):
		# Set weights to Gaussian, biases to zero
		torch.manual_seed(1337)
		torch.cuda.manual_seed(1337)
		torch.cuda.manual_seed_all(1337)
		
		#print 'init weights'
		
		for m in self.modules():
			if isinstance(m, nn.BatchNorm1d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				n = m.weight.size(1)
				m.weight.data.normal_(0, 0.01)
				#m.bias.data.zero_() + 1
				m.bias.data = torch.ones(m.bias.data.size())
