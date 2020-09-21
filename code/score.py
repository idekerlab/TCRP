import numpy as np
import torch
from torch.autograd import Variable
from utils import *
from scipy import stats

# Helper methods for evaluating a classification network

def forward_pass(net, in_, target, weights=None):
	# Forward in_ through the net, return loss and output

	input_var = Variable(in_).cuda()
	target_var = Variable(target).cuda()
	
	# Second output is hidden
	out, _ = net.net_forward(input_var, weights)

	# Here loss is MSE
	loss = net.loss_fn(out, target_var)
	
	return loss, out

def evaluate_new_PDX(net, loader, train_flag, weights=None):
	#evaluate the net on the data in the loader
	net.eval()

	test_predict = torch.zeros(0,0).cuda()
	test_label = torch.zeros(0,0).cuda()
	cat_test_label = torch.zeros(0,0).cuda()
	total_loss = 0

	for i, (in_, target, cat_target) in enumerate(loader):

		input_var = Variable(in_)
		target_var = Variable(target)

		# Second output is hidden
		out, _ = net.net_forward(input_var, weights)

		# Here loss is MSE
		l = net.loss_fn(out, target_var)

		test_predict = torch.cat([test_predict, out.data], 0)
		test_label = torch.cat([test_label, target_var.data], 0)
		cat_test_label = torch.cat([cat_test_label, cat_target], 0)

		#print l.data.cpu().numpy().shape
		total_loss += l.data.cpu().numpy()

	if test_predict.size()[0] <= 1:
		pear_corr = -1
	else:
		pear_corr = pearson_corr(test_predict, test_label)

	rho, pval = stats.spearmanr( test_predict.data.cpu().numpy() ,test_label.data.cpu().numpy() )

	predict_data = {}
	predict_data['test_predict'] = test_predict.data.cpu().numpy()
	predict_data['test_label'] = test_label.data.cpu().numpy()
	predict_data['cat_test_label'] = cat_test_label.cpu().numpy()

	return float(total_loss) / test_label.size()[0], pear_corr, rho, predict_data

def evaluate_new(net, loader, train_flag, weights=None):
	#evaluate the net on the data in the loader
	net.eval()

	test_predict = torch.zeros(0,0).cuda()
	test_label = torch.zeros(0,0).cuda()
	total_loss = 0

	for i, (in_, target) in enumerate(loader):

		input_var = Variable(in_)
		target_var = Variable(target)

		# Second output is hidden
		out, _ = net.net_forward(input_var, weights)

		# Here loss is MSE
		l = net.loss_fn(out, target_var)

		test_predict = torch.cat([test_predict, out.data], 0)
		test_label = torch.cat([test_label, target_var.data], 0)

		#print l.data.cpu().numpy().shape
		total_loss += l.data.cpu().numpy()

	if test_predict.size()[0] <= 1:
		pear_corr = -1
	else:
		pear_corr = pearson_corr(test_predict, test_label)

	rho, pval = stats.spearmanr( test_predict.data.cpu().numpy() ,test_label.data.cpu().numpy() )

	return float(total_loss) / test_label.size()[0], pear_corr, rho, test_predict, test_label

def evaluate_cv(net, loader, weights=None):
	#evaluate the net on the data in the loader
	net.eval()

	test_predict = torch.zeros(0,0).cuda()
	test_label = torch.zeros(0,0).cuda()
	total_loss = 0
	#print 'In size evaluate'

	for i, (in_, target) in enumerate(loader):

		input_var = Variable(in_)
		target_var = Variable(target)

		# Second output is hidden
		out, _ = net.net_forward(input_var, weights)

		# Here loss is MSE
		l = net.loss_fn(out, target_var)

		test_predict = torch.cat([test_predict, out.data], 0)
		test_label = torch.cat([test_label, target_var.data], 0)

		total_loss += l.data.cpu().numpy()

	pear_corr = pearson_corr(test_predict, test_label)
	return float(total_loss) / test_label.size()[0], pear_corr, test_predict

def evaluate(net, loader, train_flag, weights=None):
	#evaluate the net on the data in the loader
	net.eval()
	
	test_predict = torch.zeros(0,0).cuda()
	test_label = torch.zeros(0,0).cuda()	
	total_loss = 0
	#print 'In size evaluate'

	for i, (in_, target) in enumerate(loader):
	
		input_var = Variable(in_)
		target_var = Variable(target)

		# Second output is hidden
		out, _ = net.net_forward(input_var, weights)

		# Here loss is MSE
		l = net.loss_fn(out, target_var)

		test_predict = torch.cat([test_predict, out.data], 0)
		test_label = torch.cat([test_label, target_var.data], 0)
		
		#total_loss += l.data.cpu().numpy()[0]
		total_loss += l.data.cpu().numpy()
		#aa, bb = out.data.cpu().numpy(), target_var.data.cpu().numpy()
		#print np.mean(np.square(aa - bb)), l.data.cpu().numpy()[0]

	if test_predict.size()[0] <= 3:
		pear_corr = -1
	else:
		pear_corr = pearson_corr(test_predict, test_label)
	
	#print test_predict.cpu().numpy()[:,0]
	#print test_label.cpu().numpy()[:,0]
	#print 'finish evaluate'
	'''
	test_predict = test_predict.cpu().numpy()
	for i in range(test_predict.shape[0]):
		print test_predict[i,0],
	print ''
	
	test_label = test_label.cpu().numpy()
	for i in range(test_label.shape[0]):
		print test_label[i,0],
	print ''
	'''
	return float(total_loss) / test_label.size()[0], pear_corr
