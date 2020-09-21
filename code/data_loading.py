import numpy as np
import random
import copy
import torch
import torch.utils.data as du

'''
def get_unseen_data_loader( feature, label, cat_label, K ):

	num_sample = feature.shape[0]
	index_list = np.random.permutation( num_sample )
	
	train_index_list = index_list[0:K]
	test_index_list = index_list[K:]

	train_feature = torch.FloatTensor( feature[train_index_list,:] )
	train_label = torch.FloatTensor( label[train_index_list,:] )

	test_feature = torch.FloatTensor( feature[test_index_list,:] )
	test_label = torch.FloatTensor( label[test_index_list,:] )
	cat_test_label = torch.FloatTensor( cat_label[test_index_list,:] )

	train_dataset = du.TensorDataset( train_feature.cuda(), train_label.cuda() )
	train_loader = du.DataLoader( train_dataset, batch_size=train_feature.size(0))

	test_dataset = du.TensorDataset( test_feature.cuda(), test_label.cuda(), cat_test_label.cuda() )
	test_loader = du.DataLoader( test_dataset, batch_size=test_feature.size(0))

	return train_loader, test_loader
'''
def get_observed_data_loader(feature, label, tissue_index_list, K, batch_size, tissue_num):

	index_list = copy.deepcopy(tissue_index_list)
	train_sampled_index_list, test_sampled_index_list  = [], []

	#print 'tissue index list..', len(tissue_index_list)

	random_tissue_index = np.random.permutation( len(tissue_index_list) ) 
	
	train_tissue_index_list = random_tissue_index[0:tissue_num]
	#train_tissue_index_list = random_tissue_index[0:-1]

	test_tissue_index_list = random_tissue_index[tissue_num:tissue_num*2]
	#test_tissue_index_list = [random_tissue_index[-1]]

	#print train_tissue_index_list
	#print test_tissue_index_list
	
	for tissue_index in train_tissue_index_list:

		sub_list = index_list[tissue_index]
		random.shuffle(sub_list)

		train_sampled_index_list +=	sub_list[0:K]

	for tissue_index in test_tissue_index_list:

		sub_list = index_list[tissue_index]
		random.shuffle(sub_list)

		test_sampled_index_list += sub_list[0:K]
		
	random.shuffle( train_sampled_index_list )
	random.shuffle( test_sampled_index_list )

	#print '===', train_sampled_index_list
	train_feature = torch.FloatTensor( feature[train_sampled_index_list,:] )
	train_label = torch.FloatTensor( label[train_sampled_index_list,:] )

	dataset = du.TensorDataset(train_feature, train_label)
	loader = du.DataLoader( dataset, batch_size=batch_size, pin_memory=True )

	train_data_list = []
	for batch_feature, batch_label in loader:
		if batch_feature.size()[0] == 1:
			continue
		train_data_list.append( (batch_feature.cuda(), batch_label.cuda()) )

	#print '===', test_sampled_index_list,feature.shape
	test_feature = torch.FloatTensor( feature[test_sampled_index_list,:] )
	test_label = torch.FloatTensor( label[test_sampled_index_list,:] )

	dataset = du.TensorDataset( test_feature, test_label )
	loader = du.DataLoader( dataset, batch_size=batch_size, pin_memory=True )

	test_data_list = []
	for batch_feature, batch_label in loader:
		if batch_feature.size()[0] == 1:
			continue
		test_data_list.append( (batch_feature.cuda(), batch_label.cuda()) )

	return train_data_list, test_data_list

def get_observed_data_loader2(feature, label, tissue_index_list, K, batch_size):
	
	index_list = copy.deepcopy(tissue_index_list)
	train_sampled_index_list, test_sampled_index_list  = [], []

	for index, sub_list in enumerate(index_list):

		random.shuffle(sub_list)

		if 2*K < len(sub_list):
			train_sampled_index_list += sub_list[0:K]
			test_sampled_index_list += sub_list[K:2*K]

		elif K < len(sub_list):
			train_sampled_index_list += sub_list[0:K]
			random.shuffle(sub_list)
			test_sampled_index_list += sub_list[0:K]

		else:
			train_sampled_index_list += sub_list
			test_sampled_index_list += sub_list

	random.shuffle( train_sampled_index_list )
	random.shuffle( test_sampled_index_list )
	
	train_feature = torch.FloatTensor( feature[train_sampled_index_list,:] )
	train_label = torch.FloatTensor( label[train_sampled_index_list,:] )
	
	dataset = du.TensorDataset(train_feature, train_label)	
	loader = du.DataLoader( dataset, batch_size=batch_size, pin_memory=True )
	
	train_data_list = []
	for batch_feature, batch_label in loader:
		train_data_list.append( (batch_feature.cuda(), batch_label.cuda()) )

	test_feature = torch.FloatTensor( feature[test_sampled_index_list,:] )
	test_label = torch.FloatTensor( label[test_sampled_index_list,:] )
	
	dataset = du.TensorDataset( test_feature, test_label )
	loader = du.DataLoader( dataset, batch_size=batch_size, pin_memory=True )

	test_data_list = []
	for batch_feature, batch_label in loader:
		test_data_list.append( (batch_feature.cuda(), batch_label.cuda()) )

	return train_data_list, test_data_list

def load_unseen_data_loader(train_index_file, test_index_file, feature, label, K, trial, batch_size=1):

	train_index_list = np.load( train_index_file )
	test_index_list = np.load( test_index_file )

	train_feature = torch.FloatTensor( feature[train_index_list,:] )
	train_label = torch.FloatTensor( label[train_index_list,] )

	test_feature = torch.FloatTensor( feature[test_index_list,:] )
	test_label = torch.FloatTensor( label[test_index_list,] )

	train_dataset = du.TensorDataset( train_feature, train_label )
	test_dataset = du.TensorDataset( test_feature, test_label )

	train_loader = du.DataLoader(train_dataset, batch_size=1)
	train_data_list = []
	for batch_feature, batch_label in train_loader:
		train_data_list.append((batch_feature.cuda(), batch_label.cuda()))

	test_loader = du.DataLoader(test_dataset, batch_size=batch_size)
	test_data_list = []
	for batch_feature, batch_label in test_loader:
		test_data_list.append((batch_feature.cuda(), batch_label.cuda()))

	return train_data_list, test_data_list

def get_unseen_data_loader(feature, label, K, batch_size=1):

	index_list = np.random.permutation(feature.shape[0])

	train_index_list = index_list[0:K]
	test_index_list = index_list[K:]

	train_feature = torch.FloatTensor( feature[train_index_list,:] )
	train_label = torch.FloatTensor( label[train_index_list,] )

	test_feature = torch.FloatTensor( feature[test_index_list,:] )
	test_label = torch.FloatTensor( label[test_index_list,] )

	train_dataset = du.TensorDataset( train_feature, train_label )
	test_dataset = du.TensorDataset( test_feature, test_label )

	train_loader = du.DataLoader(train_dataset, batch_size=batch_size)
	train_data_list = []
	for batch_feature, batch_label in train_loader:
		train_data_list.append((batch_feature.cuda(), batch_label.cuda()))
	
	test_loader = du.DataLoader(test_dataset, batch_size=batch_size)
	test_data_list = []
	for batch_feature, batch_label in test_loader:
		test_data_list.append((batch_feature.cuda(), batch_label.cuda()))

	return train_data_list, test_data_list
