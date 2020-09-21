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
from meta_learner_cv import *

# Training settings
# This model uses expresion + somatic mutations + CNV as features
# It applies the cross validation framework proposed by Siamese Network
parser = argparse.ArgumentParser()
work_dic = '/share/data/jinbodata/siqi/mut_exp_cnv_data/challenge_GDSC_PDTC/'
#work_dic = '/share/data/jinbodata/siqi/mut_exp_cnv_data/challenge_GDSC_PDX/'
#data_dic = '/share/data/jinbodata/siqi/mut_exp_cnv_data/challenge_4/tissue_test_data/'

parser.add_argument('--tissue', type=str, default='UPPER_AERODIGESTIVE_TRACT', help='Testing tissue, using the rest tissues for training')
parser.add_argument('--drug', type=str, default='AC220', help='Treated drug')
parser.add_argument('--gene', type=str, default='TP53', help='Knockout genes')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=100, help='Random seed.')
parser.add_argument('--K', type=int, default=10, help='Perform K shot learning')
parser.add_argument('--meta_batch_size', type=int, default=32, help='Meta-learning batch size, i.e. how many different tasks we need sample')
parser.add_argument('--inner_batch_size', type=int, default=10, help='Batch size for each individual learning job')
parser.add_argument('--num_updates', type=int, default=30, help='Number of training epochs')
parser.add_argument('--num_inner_updates', type=int, default=1, help='Initial learning rate')
parser.add_argument('--num_trials', type=int, default=50, help='Number of trials for unseen tissue')
parser.add_argument('--hidden', type=int, default=60, help='Number of hidden units of NN for single task')
parser.add_argument('--patience', type=int, default=3, help='Patience')
parser.add_argument('--tissue_list', type=str, default=work_dic + 'cell_line_data/tissue_cell_line_list.pkl', help='Cell line list for different tissues')
parser.add_argument('--log_folder', type=str, default=work_dic+'Log/', help='Log folder')

parser.add_argument('--meta_lr', type=float, default=0.001, help='Learning rate for meta-learning update')
parser.add_argument('--inner_lr', type=float, default=0.001, help='Learning rate for ')
parser.add_argument('--tissue_num', type=int, default=12, help='Tissue number evolved in the inner update')
parser.add_argument('--layer', type=int, default=1, help='Number of layers of NN for single task')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

drug = args.drug
K = args.K
num_trials = args.num_trials
meta_batch_size = args.meta_batch_size
inner_batch_size = args.inner_batch_size
num_updates = args.num_updates
num_inner_updates = args.num_inner_updates
patience = args.patience

#random.seed(args.seed)
#np.random.seed(args.seed)
#torch.manual_seed(args.seed)

tissue_list = work_dic + drug + '_tissue_map.pkl' 
print tissue_list
# Load tissue cell line mapping
with open(tissue_list, 'rb') as f:
	tissue_map = pickle.load(f)

# Load data
cv_feature_list, cv_label_list, meta_tissue_index_list, test_feature_list, test_label_list, test_tissue_list  = load_data_cell_line(tissue_map, drug, K)

#PDX_feature, PDX_label = load_data_PDX( args.drug )

layer, hidden, meta_lr, inner_lr, tissue_num = args.layer, args.hidden, args.meta_lr, args.inner_lr, args.tissue_num

best_fewshot_train_corr_list = []
best_fewshot_test_corr_list = []

for i, test_tissue in enumerate(test_tissue_list):

	#print 'Validate tissue', test_tissue, cv_feature_list[i].shape

	meta_dataset = dataset(cv_feature_list[i], cv_label_list[i])
	test_dataset = dataset(test_feature_list[i], test_label_list[i])

	meta_learner = MetaLearner( meta_dataset, test_dataset, K, meta_lr, inner_lr, layer, hidden, tissue_num, meta_batch_size, inner_batch_size, num_updates, num_inner_updates, meta_tissue_index_list[i], num_trials )

	best_fewshot_train_loss, best_fewshot_train_corr, best_fewshot_test_loss, best_fewshot_test_corr = meta_learner.train()

	print test_tissue, 'best few-shot train loss', best_fewshot_train_loss, 'best few-shot train corr', best_fewshot_train_corr 
	print 'best few-shot test loss', best_fewshot_test_loss, 'best few-shot test corr', best_fewshot_test_corr

	if best_fewshot_train_corr != -1:
		best_fewshot_train_corr_list.append(best_fewshot_train_corr)

	if best_fewshot_test_corr != -1:
		best_fewshot_test_corr_list.append( best_fewshot_test_corr )

print 'Avg_test corr', np.asarray(best_fewshot_train_corr_list).mean(), np.asarray(best_fewshot_test_corr_list).mean()

