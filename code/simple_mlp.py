import sys
import os
import numpy as np
import torch
import torch.utils.data as du
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from utils import *

class simple_mlp(nn.Module):

	def __init__(self, feature_dim, layer, hidden):
		
		super(simple_mlp, self).__init__()

		self.layer = layer

		self.linear1 = nn.Linear(feature_dim, hidden)
		#self.bn1= nn.BatchNorm1d( hidden )		

		if layer == 2:
			self.linear2 = nn.Linear(hidden, hidden)
			#self.bn2 = nn.BatchNorm1d( hidden )

		self.linear3 = nn.Linear(hidden, 1)	
		#self.bn3 = nn.BatchNorm1d( 1 )

	def forward(self, x, weights = None):

		#hidden = F.tanh( self.bn1( self.linear1(x) ) )
		#hidden = self.bn1 ( F.tanh( self.linear1(x) ) )
		#hidden = F.relu( self.bn1 ( self.linear1(x) ) )
		hidden = F.relu( self.linear1(x) )

		if self.layer == 2:
			#hidden = F.tanh( self.bn2( self.linear2(hidden) ) )
			#hidden = self.bn2( F.tanh( self.linear2(hidden) ) )
			#hidden = F.relu( self.bn2( self.linear2(hidden) ) )
			hidden = F.relu( self.linear2(hidden) )

		#out = F.tanh( self.linear3(hidden) )
		out = self.linear3(hidden)
		#out = self.bn3(out)

		return out, hidden
