import numpy as np
import scipy.sparse as sp
import torch
import sys
from scipy import stats
from sklearn import preprocessing
import torch.utils.data as du

def cut_data_into_pieces(feature, label):

	dataset = du.TensorDataset( torch.FloatTensor(feature), torch.FloatTensor(label) )
	data_loader = du.DataLoader( dataset, batch_size=200 )
	data_list = []

	for batch_feature, batch_label in data_loader:
		data_list.append((batch_feature.cuda(), batch_label.cuda()))

	return data_list

def encode_onehot(labels):
	classes = set(labels)
	classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
	labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
	return labels_onehot

def pearson_corr(x, y):
	
	xx = x - torch.mean(x)
	yy = y - torch.mean(y)

	norm_x = torch.norm(xx, 2)
	norm_y = torch.norm(yy,2)
	
	if norm_x ==0 or norm_y ==0:
		return 0
	else:
		return torch.sum(xx*yy) / ( norm_x * norm_y )

def load_merged_data(tissue_map, drug, path='/home-nfs/wangsheng/GDSC_PDX_WP_feature/'):

	#Load dataset
	print 'Loading feature and label files...'

	feature_list, label_list, tissue2id, tissue_index_list = [], [], [], []
	sample2tissue = {}
	sample_num = 0

	for tissue, cell_line_list in tissue_map.items():

		tissue2id.append( tissue )

		path_suffix = path + drug + '/' + tissue + '_' + drug

		feature_list.append( np.load(path_suffix + '_feature.npy') )
		label_list.append( np.load(path_suffix + '_label.npy').reshape(-1,1) )

		sub_list = []
		for i in range(len(cell_line_list)):
			sub_list.append( i+sample_num )
			sample2tissue[i+sample_num] = tissue

		tissue_index_list.append( sub_list )

		sample_num += len(cell_line_list)

		#print tissue, 'contains', len(cell_line_list), 'cell lines, accumulate', sample_num, 'cell lines', feature_list[-1].shape

	train_feature = np.concatenate(feature_list, axis=0)
	train_label = np.concatenate(label_list, axis=0)

	#scaler = preprocessing.StandardScaler().fit(train_feature)
	#train_feature = scaler.transform(train_feature)

	#path_suffix = path + drug + '/' + 'BREAST_' + drug
	#drug_test_feature = np.load(path_suffix + '_feature.npy')
	#drug_test_label = np.load(path_suffix + '_label.npy').reshape(-1,1)
	#test_feature = scaler.transform(drug_test_feature)

	path_suffix = path + drug + '/' + 'PDX_' + drug

	PDX_feature = np.load( path_suffix + '_feature.npy' )
	PDX_label = np.load( path_suffix + '_BestAvgResponse_label.npy' ).reshape(-1,1)
	#PDX_cat_label = np.load( path_suffix + '_ResponseCategory_label.npy').reshape(-1,1)
	#PDX_cat_label = np.load( path_suffix + '_TimeToDouble_label.npy').reshape(-1,1)

	#scaler = preprocessing.StandardScaler().fit(PDX_feature)
	#PDX_feature = scaler.transform(PDX_feature)

	#print 'ddddd', np.argwhere(np.isnan(PDX_label))
	#print PDX_label
	scaler2 = preprocessing.StandardScaler().fit(PDX_label)
	PDX_label = scaler2.transform(PDX_label)

	print 'Cell line feature dimension', train_feature.shape, train_label.shape
	print 'PDX feature dimension', PDX_feature.shape, PDX_label.shape

	return train_feature, train_label, tissue_index_list, PDX_feature, PDX_label, sample2tissue

# need to edit this function
#def load_data_cell_line(tissue_map, drug, K, path='/share/data/jinbodata/siqi/mut_exp_cnv_data/challenge_GDSC_PDTC/GDSC_PPI_feature/'):
#def load_data_cell_line(tissue_map, drug, K, path='/share/data/jinbodata/siqi/mut_exp_cnv_data/challenge_GDSC_PDTC/GDSC_won-parafac_feature/'):
def load_data_cell_line(tissue_map, drug, K, path='/home-nfs/wangsheng/GDSC_PDX_WP_feature/'):
	#Load dataset
	print 'Loading feature and label files...'
	feature_map, label_map = {}, {}
	all_tissue_feature_list = []

	for tissue, cell_line_list in tissue_map.items():

		path_suffix = path + drug + '/' + tissue + '_' + drug
		tissue_feature = np.load(path_suffix + '_feature.npy')
		tissue_label = np.load(path_suffix + '_label.npy').reshape(-1,1)

		feature_map[ tissue + '_' + drug ] = tissue_feature
		label_map[ tissue + '_' + drug ] = tissue_label

		all_tissue_feature_list.append(tissue_feature)			

	all_tissue_train_feature = np.concatenate(all_tissue_feature_list, axis=0)
	scaler = preprocessing.StandardScaler().fit(all_tissue_train_feature)
	#train_feature = scaler.transform(all_tissue_train_feature)
	print 'all tissue shape', all_tissue_train_feature.shape

	cv_feature_list, cv_label_list, cv_tissue_index_list = [], [], []
	vali_feature_list, vali_label_list, vali_tissue_list = [], [], []

	for temp_tissue, temp_cell_line_list in tissue_map.items():
		
		if len(temp_cell_line_list) == 0:
			continue
		feature_map[ temp_tissue+'_'+drug ] = scaler.transform( feature_map[ temp_tissue+'_'+drug ] )
		#print temp_tissue, feature_map[ temp_tissue+'_'+drug ].shape

	for temp_tissue, temp_cell_line_list in tissue_map.items():

		if len(temp_cell_line_list) <= K:
			continue
		else:
			vali_feature_list.append( feature_map[ temp_tissue+'_'+drug ] )
			vali_label_list.append( label_map[ temp_tissue+'_'+drug ] )
			vali_tissue_list.append( temp_tissue )

		feature_list, label_list, tissue_index_list = [], [], []
		sample_num = 0

		for tissue, cell_line_list in tissue_map.items():

			if tissue == temp_tissue:
				continue

			feature_list.append( feature_map[ tissue + '_' + drug ] )
			label_list.append( label_map[ tissue + '_' + drug ] )

			sub_list = []
			for i in range( len(cell_line_list) ):
				sub_list.append( i+sample_num )

			tissue_index_list.append( sub_list )
			sample_num += len( cell_line_list )

		train_feature = np.concatenate(feature_list, axis=0)
		train_label = np.concatenate(label_list, axis=0)

		cv_feature_list.append( train_feature )
		cv_label_list.append( train_label )
		cv_tissue_index_list.append( tissue_index_list )

	print 'Cross validation', len(cv_feature_list), len(cv_label_list), len(vali_feature_list), len(vali_label_list)
	print 'Vali dimension', cv_feature_list[0].shape, vali_feature_list[0].shape, vali_label_list[0].shape

	#return train_feature, train_label, tissue_index_list, drug_test_feature, drug_test_label, sample2tissue
	return cv_feature_list, cv_label_list, cv_tissue_index_list, vali_feature_list, vali_label_list, vali_tissue_list

#def load_data_PDTC(drug, path='/share/data/jinbodata/siqi/mut_exp_cnv_data/challenge_GDSC_PDTC/GDSC_PPI_feature/'):
#def load_data_PDTC(drug, path='/share/data/jinbodata/siqi/mut_exp_cnv_data/challenge_GDSC_PDTC/GDSC_won-parafac_feature/'):
def load_data_PDTC(drug, path='/home-nfs/wangsheng/challenge_GDSC_PDTC/GDSC_PPI_feature/'):
	#path_suffix = path + drug + '/' + 'PDTC_' + drug
	path_suffix = path + '/' + 'PDTC_' + drug

	PDTC_feature = np.load( path_suffix + '_feature.npy' )
	PDTC_label = np.load( path_suffix + '_label.npy' ).reshape(-1,1)

	scaler = preprocessing.StandardScaler().fit(PDTC_feature)
	PDTC_feature = scaler.transform(PDTC_feature)

	#print 'ddddd', np.argwhere(np.isnan(PDX_label))
	#print PDX_label
	#scaler2 = preprocessing.StandardScaler().fit(PDTC_label)
	#PDTC_label = scaler2.transform(PDTC_label)

	print 'PDTC feature dimension', PDTC_feature.shape, PDTC_label.shape

	#return train_feature, train_label, tissue_index_list, drug_test_feature, drug_test_label, sample2tissue
	return PDTC_feature, PDTC_label

#def load_data_PDX(drug, path='/share/data/jinbodata/siqi/mut_exp_cnv_data/challenge_GDSC_PDTC/GDSC_PPI_feature/'):
#def load_data_PDX(drug, scaler, path='/home-nfs/siqi/CancerDrugPDX/challenge_GDSC_PDX/GDSC_WON_feature/'):
def load_data_PDX(drug, scaler, path='/home-nfs/wangsheng/GDSC_PDX_WP_feature/'):
	path_suffix = path + drug + '/' + 'PDX_' + drug

	PDX_feature = np.load( path_suffix + '_feature.npy' )
	PDX_label = np.load( path_suffix + '_BestAvgResponse_label.npy' ).reshape(-1,1)
	#PDX_cat_label = np.load( path_suffix + '_ResponseCategory_label.npy').reshape(-1,1)
	PDX_cat_label = np.load( path_suffix + '_TimeToDouble_label.npy').reshape(-1,1)

	print PDX_feature.shape, '===='
	scaler = preprocessing.StandardScaler().fit(PDX_feature)
	PDX_feature = scaler.transform(PDX_feature)

	#print 'ddddd', np.argwhere(np.isnan(PDX_label))
	#print PDX_label
	scaler2 = preprocessing.StandardScaler().fit(PDX_label)
	PDX_label = scaler2.transform(PDX_label)

	print 'PDX feature dimension', PDX_feature.shape, PDX_label.shape

	#return train_feature, train_label, tissue_index_list, drug_test_feature, drug_test_label, sample2tissue
	return PDX_feature, PDX_label, PDX_cat_label

#def load_data(tissue_map, drug, K, path='/share/data/jinbodata/siqi/mut_exp_cnv_data/challenge_GDSC_PDTC/GDSC_PPI_feature/'):
#def load_data(tissue_map, drug, K, path='/share/data/jinbodata/siqi/mut_exp_cnv_data/challenge_GDSC_PDTC/GDSC_won-parafac_feature/'):
#def load_data(tissue_map, drug, K, path='/home-nfs/siqi/CancerDrugPDX/challenge_GDSC_PDX/GDSC_WON_feature/'):
def load_data(tissue_map, drug, K, path='/home-nfs/wangsheng/challenge_GDSC_PDTC/GDSC_PPI_feature/'):
	#Load dataset
	print 'Loading feature and label files...'
	
	feature_list, label_list, tissue_index_list = [], [], []
	sample_num = 0

	for tissue, cell_line_list in tissue_map.items():

		if len(cell_line_list) == 0:
			continue

		#path_suffix = path + drug + '/' + tissue + '_' + drug
		path_suffix = path + '/' + tissue + '_' + drug	
	
		feature_list.append( np.load(path_suffix + '_feature.npy') )
		label_list.append( np.load(path_suffix + '_label.npy').reshape(-1,1) )

		sub_list = []
		for i in range(len(cell_line_list)):
			sub_list.append( i+sample_num )

		tissue_index_list.append(sub_list)
		sample_num += len(cell_line_list)
		#print tissue, 'contains', len(cell_line_list), 'cell lines, accumulate', sample_num, 'cell lines'
		#print np.mean(feature_list[-1])
		#print tissue, sub_list

	train_feature = np.concatenate(feature_list, axis=0)
	train_label = np.concatenate(label_list, axis=0)

	scaler = preprocessing.StandardScaler().fit(train_feature)
	train_feature = scaler.transform(train_feature) 

	print 'Drug feature dimension', train_feature.shape, train_label.shape

	return train_feature, train_label, tissue_index_list
	#return train_feature, train_label, tissue_index_list, scaler

def normalize_adj(mx):
	"""Row-normalize sparse matrix"""
	rowsum = np.array(mx.sum(1))
	r_inv_sqrt = np.power(rowsum, -0.5).flatten()
	r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
	r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
	return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

def normalize_features(mx):
	"""Row-normalize sparse matrix"""
	rowsum = np.array(mx.sum(1))
	r_inv = np.power(rowsum, -1).flatten()
	r_inv[np.isinf(r_inv)] = 0.
	r_mat_inv = sp.diags(r_inv)
	mx = r_mat_inv.dot(mx)
	return mx


def accuracy(output, labels):
	preds = output.max(1)[1].type_as(labels)
	correct = preds.eq(labels).double()
	correct = correct.sum()
	return correct / len(labels)

