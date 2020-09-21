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
# This model uses expresion + somatic mutations as features
# It applies the cross validation framework proposed by Siamese Network
parser = argparse.ArgumentParser()

parser.add_argument('--feature_dic', type=str, default='/home-nfs/siqi/TCRP/data/Sorafenib/', help='Feature folder')
parser.add_argument('--model_dic', type=str, default='/cellar/users/majianzhu/TCRP/models/', help='Feature folder')
parser.add_argument('--drug', type=str, default='AC220', help='Treated drug')
parser.add_argument('--seed', type=int, default=19, help='Random seed.')
parser.add_argument('--K', type=int, default=10, help='Perform K shot learning')
parser.add_argument('--meta_batch_size', type=int, default=32, help='Meta-learning batch size, i.e. how many different tasks we need sample')
parser.add_argument('--inner_batch_size', type=int, default=10, help='Batch size for each individual learning job')
parser.add_argument('--num_updates', type=int, default=10, help='Number of training epochs')
parser.add_argument('--num_inner_updates', type=int, default=1, help='Initial learning rate')
parser.add_argument('--num_out_updates', type=int, default=20, help='Final learning rate')
parser.add_argument('--num_trials', type=int, default=50, help='Number of trials for unseen tissue')
parser.add_argument('--hidden', type=int, default=60, help='Number of hidden units of NN for single task')
parser.add_argument('--tissue_list', type=str, default='/home-nfs/siqi/TCRP/data/Sorafenib_tissue_map.pkl', help='Cell line list for different tissues, used for defining meta-tasks in the meta-learning phase')
parser.add_argument('--meta_lr', type=float, default=0.001, help='Learning rate for meta-learning update')
parser.add_argument('--inner_lr', type=float, default=0.001, help='Learning rate for ')
parser.add_argument('--tissue_num', type=int, default=12, help='Tissue number evolved in the inner update')
parser.add_argument('--layer', type=int, default=1, help='Number of layers of NN for single task')

args = parser.parse_args()

feature_dic = args.feature_dic
drug = args.drug
K = args.K
num_trials = args.num_trials
meta_batch_size = args.meta_batch_size
inner_batch_size = args.inner_batch_size
num_updates = args.num_updates
num_inner_updates = args.num_inner_updates

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

tissue_list = args.tissue_list
print tissue_list
# Load tissue cell line mapping
with open(tissue_list, 'rb') as f:
	tissue_map = pickle.load(f)

model_dic = args.model_dic + 'MODELS/' + drug + '/'

if not os.path.exists(model_dic):
	mkdir_cmd = 'mkdir -p ' + model_dic
	os.system(mkdir_cmd) 
# Load data
#cv_feature_list, cv_label_list, meta_tissue_index_list, test_feature_list, test_label_list, test_tissue_list  = load_data_cell_line(tissue_map, drug, K)
train_feature, train_label, tissue_index_list = load_data(tissue_map, drug, K, path = feature_dic)

PDTC_feature, PDTC_label = load_data_PDTC( args.drug, path = feature_dic )
#PDX_feature, PDX_label = load_data_PDX( args.drug )
#best_param = (2,args.hidden,0.001,0.001,12)

layer, hidden, meta_lr, inner_lr, tissue_num = args.layer, args.hidden, args.meta_lr, args.inner_lr, args.tissue_num

meta_dataset = dataset(train_feature, train_label)
test_dataset = dataset(PDTC_feature, PDTC_label)

best_train_loss_test_corr_list, best_train_corr_test_corr_list, best_train_corr_test_scorr_list, best_train_scorr_test_scorr_list = [], [], [], []

for i in range(num_trials):
	meta_learner = MetaLearner( meta_dataset, test_dataset, K, meta_lr, inner_lr, layer, hidden, tissue_num, meta_batch_size, inner_batch_size, num_updates, num_inner_updates, tissue_index_list, num_trials )

	best_train_loss_test_corr, best_train_corr_test_corr, best_train_corr_test_scorr, best_train_scorr_test_scorr, best_model = meta_learner.train()
	best_train_loss_test_corr_list.append(best_train_loss_test_corr)
	best_train_corr_test_corr_list.append(best_train_corr_test_corr)
	best_train_corr_test_scorr_list.append(best_train_corr_test_scorr)
	best_train_scorr_test_scorr_list.append(best_train_scorr_test_scorr)

    # Please uncomment this line to save your pre-train models
	#torch.save(best_model, model_dic + '/model_'+str(K)+'_trail_' + str(i))

a = np.asarray(best_train_loss_test_corr_list).mean()
b = np.asarray(best_train_corr_test_corr_list).mean()
c = np.asarray(best_train_corr_test_scorr_list).mean()
d = np.asarray(best_train_scorr_test_scorr_list).mean()

print 'PDTC best_train_loss_test_corr:', float('%.3f'%a), 'best_train_corr_test_corr', float('%.3f'%b), 'best_train_corr_test_scorr', float('%.3f'%c), 'best_train_scorr_test_scorr', float('%.3f'%d)

