##this version use 1M subset for adjusting model strcture

#import files
from __future__ import division

import theano
from pymongo import MongoClient

import numpy as np#something
import time
import os
import sys
import argparse

import h5py # needed for save_weights, fails otherwise

from keras import backend as K 
from keras.backend import theano_backend
import theano.tensor as T
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Activation, Input, Concatenate
from keras.layers.core import Flatten, Permute, Reshape, Dropout, Lambda, RepeatVector
from keras.layers.wrappers import TimeDistributed
from keras.layers import merge, activations
from keras.layers.merge import Dot, Add
from keras.optimizers import SGD, Adam, Adadelta
from keras.layers.convolutional import Convolution1D, Convolution2D
from keras.regularizers import l2
from keras.utils.np_utils import to_categorical
from keras.utils.generic_utils import func_dump
from keras.utils.generic_utils import func_load
from keras import losses
from keras.metrics import top_k_categorical_accuracy
from tqdm import tqdm
import itertools, operator, random

# from keras import theano_backend
#import keras.engine.topology 

from keras.engine.topology import Layer
import pickle
# from rdkit import Chem, DataStructs
# from rdkit.Chem import AllChem
import os
from sklearn.metrics import roc_auc_score
from rdkit import RDLogger

# from makeit.embedding.descriptors import edits_to_vectors, oneHotVector # for testing
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from scipy import sparse,stats
import datetime

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

client = None####mongo database

# db = client['reaxys_v2']

# chemical_db = db['chemicals']
# reaction_db = db['reactions']

def set_keras_backend(backend):
	if K.backend() != backend:
		os.environ['KERAS_BACKEND'] = backend
		reload(K)
		assert K.backend() == backend
##USEFUL UTILITIES
def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    loader = np.load(filename)
    return sparse.csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])
def categorical_crossentropy_with_logits(y_true, y_pred):
	return T.nnet.categorical_crossentropy(y_pred, y_true)

def intop3(y_true,y_pred):
	return top_k_categorical_accuracy(y_true, y_pred, k=3) 

K.logical_not = lambda x: 1 - x     # tf.logical_not
K.is_nan = T.isnan
def mse_ignore_na(y_true, y_pred):
    return K.mean(K.switch(K.is_nan(y_true), 0, K.square(y_pred - y_true)))
#fingerprint based training
def build(pfp_len = 2048, rxnfp_len = 2048, c1_dim = 100, s1_dim = 100, s2_dim =100, r1_dim = 100, r2_dim =100, N_h1 = 1024, N_h2 = 100,l2v = 0, lr = 0.01):
	input_pfp = Input(shape = (pfp_len,), name = 'input_pfp')
	input_rxnfp = Input(shape = (rxnfp_len,), name = 'input_rxnfp')

	input_r1 = Input(shape = (r1_dim,), name = 'input_r1')
	input_r2 = Input(shape = (r2_dim,), name = 'input_r2')
	input_s1 = Input(shape = (s1_dim,), name = 'input_s1')
	input_s2 = Input(shape = (s2_dim,), name = 'input_s2')
	input_c1 = Input(shape = (c1_dim,), name = 'input_c1')
	

	concat_fp = Concatenate(axis = 1)([input_pfp,input_rxnfp])

	h1 = Dense(1000, activation = 'relu', kernel_regularizer = l2(l2v),name = 'fp_transform1')(concat_fp)
	# h1_dropout = Dropout(0.3)(h1)
	h2 = Dense(1000, activation = 'relu', kernel_regularizer = l2(l2v),name = 'fp_transform2')(h1)
	h2_dropout = Dropout(0.5)(h2,training=False)
	# h1 = concat_fp

	c1_h1 = Dense(N_h1, activation = 'relu', kernel_regularizer = l2(l2v), name = 'c1_h1')(h2_dropout)
	c1_h2 = Dense(N_h1, activation = 'tanh', kernel_regularizer = l2(l2v), name = 'c1_h2')(c1_h1)
	# c1_h2_dropout = Dropout(0.3)(c1_h2)
	c1_output = Dense(c1_dim, activation = "softmax",name = "c1")(c1_h2)
	c1_dense = Dense(N_h2, activation = 'relu',name = 'c1_dense')(input_c1)

	concat_fp_c1 = Concatenate(axis = -1,name = "concat_fp_c1")([h2_dropout,c1_dense])

	s1_h1 = Dense(N_h1, activation = 'relu', kernel_regularizer = l2(l2v), name = "s1_h1")(concat_fp_c1)
	s1_h2 = Dense(N_h1, activation = 'tanh', kernel_regularizer = l2(l2v), name = "s1_h2")(s1_h1)
	# s1_h2_dropout = Dropout(0.3)(s1_h2)
	s1_output = Dense(s1_dim, activation = "softmax", name = "s1")(s1_h2)
	s1_dense = Dense(N_h2, activation = 'relu',name = 's1_dense')(input_s1)
	# rgt_output = Lambda(lambda x: x / K.sum(x, axis=-1),output_shape = (rgt_dim,))(rgt_unscaled)

	concat_fp_c1_s1 = Concatenate(axis = -1,name = "concat_fp_c1_s1")([h2_dropout,c1_dense,s1_dense])
	
	s2_h1 = Dense(N_h1, activation = 'relu', kernel_regularizer = l2(l2v), name = "s2_h1")(concat_fp_c1_s1)
	s2_h2 = Dense(N_h1, activation = 'tanh', kernel_regularizer = l2(l2v), name = "s2_h2")(s2_h1)
	# s2_h2_dropout = Dropout(0.3)(s2_h2)
	s2_output = Dense(s2_dim, activation = "softmax", name = "s2")(s2_h2)	
	s2_dense = Dense(N_h2, activation = 'relu',name = 's2_dense')(input_s2)

	concat_fp_c1_s1_s2 = Concatenate(axis = -1,name = "concat_fp_c1_s1_s2")([h2_dropout,c1_dense, s1_dense, s2_dense])

	r1_h1 = Dense(N_h1, activation = 'relu', kernel_regularizer = l2(l2v), name = "r1_h1")(concat_fp_c1_s1_s2)
	r1_h2 = Dense(N_h1, activation = 'tanh', kernel_regularizer = l2(l2v), name = "r1_h2")(r1_h1)
	# r1_h2_dropout = Dropout(0.3)(r1_h2)
	r1_output = Dense(r1_dim, activation = "softmax", name = "r1")(r1_h2)
	r1_dense = Dense(N_h2, activation = 'relu',name = 'r1_dense')(input_r1)
	# rgt_output = Lambda(lambda x: x / K.sum(x, axis=-1),output_shape = (rgt_dim,))(rgt_unscaled)

	concat_fp_c1_s1_s2_r1 = Concatenate(axis = -1,name = "concat_fp_c1_s1_s2_r1")([h2_dropout,c1_dense,s1_dense,s2_dense,r1_dense])
	
	r2_h1 = Dense(N_h1, activation = 'relu', kernel_regularizer = l2(l2v), name = "r2_h1")(concat_fp_c1_s1_s2_r1)
	r2_h2 = Dense(N_h1, activation = 'tanh', kernel_regularizer = l2(l2v), name = "r2_h2")(r2_h1)
	# r2_h2_dropout = Dropout(0.3)(r2_h2)
	r2_output = Dense(r2_dim, activation = "softmax", name = "r2")(r2_h2)
	r2_dense = Dense(N_h2, activation = 'relu',name = 'r2_dense')(input_r2)	

	concat_fp_c1_s1_s2_r1_r2 = Concatenate(axis = -1,name = "concat_fp_c1_s1_s2_r1_r2")([h2_dropout,c1_dense,s1_dense,s2_dense,r1_dense,r2_dense])
	
	T_h1 = Dense(N_h1, activation = 'relu', name = "T_h1")(concat_fp_c1_s1_s2_r1_r2)
	# T_h1_dropout = Dropout(0.3)(T_h1)
	# T_h2 = Dense(N_h1, activation = 'relu', kernel_regularizer = l2(l2v), name = "T_h2")(T_h1)
	T_output = Dense(1, activation = "linear", name = "T")(T_h1)	
	#just for the purpose of shorter print message
	c1 = c1_output
	r1 = r1_output
	r2 = r2_output
	s1 = s1_output
	s2 = s2_output
	Temp = T_output
	output = [c1,r1,r2,s1,s2,Temp]
	model = Model([input_pfp,input_rxnfp,input_c1,input_r1,input_r2,input_s1,input_s2],output)
	# print(model.output_shape)

	model.count_params()
	model.summary()
	adam = Adam(lr = lr)
	sgd = SGD(lr = lr)
	model.compile(loss=[categorical_crossentropy_with_logits,categorical_crossentropy_with_logits,categorical_crossentropy_with_logits,\
		categorical_crossentropy_with_logits,categorical_crossentropy_with_logits,mse_ignore_na], loss_weights= [1,1,1, 1,1,0.0001], optimizer='adam',
		# metrics = {'c1':['acc',intop3],'s1':['acc',intop3],'s2':['acc',intop3],'r1':['acc',intop3],'r2':['acc',intop3]}
		)
	return model


def load_and_partition_data(pfp_csrmtx, rfp_csrmtx, rxn_id_list, cat_1_mtx, rgt_1_mtx, rgt_2_mtx, slv_1_mtx, slv_2_mtx, temp_list, split_ratio, batch_size):
	N_samples = pfp_csrmtx.shape[0]
	N_train = int(N_samples * split_ratio[0])
	N_val	= int(N_samples * split_ratio[1])
	N_test  = N_samples - N_train - N_val
	print('Total number of samples: {}'.format(N_samples))
	print('Training   on {}% - {}'.format(split_ratio[0]*100, N_train))
	print('Validating on {}% - {}'.format(split_ratio[1]*100, N_val))
	print('Testing    on {}% - {}'.format((1-split_ratio[1]-split_ratio[0])*100, N_test))

	#change
	return {
		'N_samples': N_samples,
		'N_train': N_train,
		#
		'train_generator': batch_data_generator(pfp_csrmtx, rfp_csrmtx, cat_1_mtx, rgt_1_mtx, rgt_2_mtx, slv_1_mtx, slv_2_mtx, temp_list,  0, N_train, batch_size),
		'train_label_generator': batch_label_generator(rxn_id_list, 0, N_train, batch_size),
		'train_nb_samples': N_train,
		#
		'val_generator': batch_data_generator(pfp_csrmtx, rfp_csrmtx, cat_1_mtx, rgt_1_mtx, rgt_2_mtx, slv_1_mtx, slv_2_mtx, temp_list,  N_train, N_train + N_val, batch_size),
		'val_label_generator': batch_label_generator(rxn_id_list, N_train, N_train + N_val, batch_size),
		'val_nb_samples': N_val,
		#
		'test_generator': batch_data_generator(pfp_csrmtx, rfp_csrmtx, cat_1_mtx, rgt_1_mtx, rgt_2_mtx, slv_1_mtx, slv_2_mtx, temp_list,  N_train + N_val, N_samples, batch_size),
		'test_label_generator': batch_label_generator(rxn_id_list, N_train + N_val, N_samples, batch_size),
		'test_nb_samples': N_test,
		#
		#
		'batch_size': batch_size,
	}


#batch data generator
def batch_data_generator(pfp_csrmtx, rfp_csrmtx, cat_1_mtx, rgt_1_mtx, rgt_2_mtx, slv_1_mtx, slv_2_mtx, temp_list,  start_at, end_at, batch_size):

	while True:
		for start_index in range(start_at, end_at,batch_size):
			end_index = min(start_index + batch_size, end_at)

			cat_1_train_batch = cat_1_mtx[start_index:end_index,:].todense()
			rgt_1_train_batch = rgt_1_mtx[start_index:end_index,:].todense()
			rgt_2_train_batch = rgt_2_mtx[start_index:end_index,:].todense()
			slv_1_train_batch = slv_1_mtx[start_index:end_index,:].todense()
			slv_2_train_batch = slv_2_mtx[start_index:end_index,:].todense()
			temp_train_batch = temp_list[start_index:end_index]
			# y_train_batch = y_train_batch>0
			# y_row_sum = y_train_batch_unscaled.sum(axis = 1)
			# y_row_sum = y_row_sum.reshape(end_index-start_index,1)
			# print(y_row_sum.shape)
			# y_train_batch = y_train_batch_unscaled / (y_row_sum)
			# y_train_batch = 2*(y_train_batch-0.5)
			# print(y_train_batch[0,:].sum( axis = 1))

			pfp_denmtx = pfp_csrmtx[start_index:end_index,:].todense()
			rfp_denmtx = rfp_csrmtx[start_index:end_index,:].todense()
			rxnfp_denmtx = pfp_denmtx - rfp_denmtx
			pfp_train_batch = np.asarray(pfp_denmtx, dtype = 'float32')
			rxnfp_train_batch = np.asarray(rxnfp_denmtx, dtype = 'float32')
			cat_1_train_batch = np.asarray(cat_1_train_batch, dtype = 'float32')
			rgt_1_train_batch = np.asarray(rgt_1_train_batch, dtype = 'float32')
			rgt_2_train_batch = np.asarray(rgt_2_train_batch, dtype = 'float32')
			slv_1_train_batch = np.asarray(slv_1_train_batch, dtype = 'float32')
			slv_2_train_batch = np.asarray(slv_2_train_batch, dtype = 'float32')
			temp_train_batch = np.asarray(temp_train_batch, dtype = 'float32')
			
			# print(pfp_train_batch[0,:])
			# print(rxnfp_train_batch[0,:])
			# print(y_train_batch[0,:])
			yield ([pfp_train_batch, rxnfp_train_batch,cat_1_train_batch, rgt_1_train_batch, rgt_2_train_batch, slv_1_train_batch, slv_2_train_batch],[cat_1_train_batch, rgt_1_train_batch, rgt_2_train_batch, slv_1_train_batch, slv_2_train_batch, temp_train_batch])


def batch_label_generator(rxn_id_list, start_at, end_at, batch_size):

	while True:
		for start_index in range(start_at, end_at,batch_size):
			end_index = min(start_index + batch_size, end_at)

			rxn_ids = rxn_id_list[start_index:end_index]

			yield (rxn_ids)

def train(model, pfp_csrmtx, rfp_csrmtx, cat_1_mtx, rgt_1_mtx, rgt_2_mtx, slv_1_mtx, slv_2_mtx, temp_list,  split_ratio, class_weight, batch_size):
	data = load_and_partition_data(pfp_csrmtx, rfp_csrmtx, rxn_id_list, cat_1_mtx, rgt_1_mtx, rgt_2_mtx, slv_1_mtx, slv_2_mtx, temp_list,  split_ratio,batch_size)

	# Add additional callbacks
	from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, ReduceLROnPlateau
	reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
              patience=2, min_lr=0.00001)
	callbacks = [
		ModelCheckpoint(BEST_WEIGHTS_FPATH, save_weights_only = True, save_best_only = True), # save every epoch
		EarlyStopping(patience = 2),
		CSVLogger(LOG_FPATH),
		# reduce_lr
		]

	try:
		hist = model.fit_generator(data['train_generator'], 
			verbose = 1,
			validation_data = data['val_generator'],
			steps_per_epoch = np.ceil(data['train_nb_samples']/data['batch_size']),
			epochs = nb_epoch, 
			callbacks = callbacks,
			validation_steps = np.ceil(data['val_nb_samples']/data['batch_size']),
			class_weight = class_weight,
		)

	except KeyboardInterrupt:
		print('Stopped training early!')

#test model
## use 5 reagents and 3 solvents to generate top 15 combos
## that means use top 1 prediction for cat, slv an rgt 2, and temp
def test(model, pfp_csrmtx, rfp_csrmtx, cat_1_mtx, rgt_1_mtx, rgt_2_mtx, slv_1_mtx, slv_2_mtx, temp_list,  split_ratio, class_weight, c1_dim, r1_dim, r2_dim, s1_dim, s2_dim, rgt_le, slv_le, cat_le, batch_size):
	data = load_and_partition_data(pfp_csrmtx, rfp_csrmtx, rxn_id_list, cat_1_mtx, rgt_1_mtx, rgt_2_mtx, slv_1_mtx, slv_2_mtx, temp_list,  split_ratio, batch_size)
	TEST_FPATH = "/home/hanyug/Make-It/makeit/context_pred/results/c_s_r_fullset/test_results_cn.dat"
	fid = open(TEST_FPATH, 'w')
	def test_on_set(fid, dataset, data_generator, label_generator, num_batches, c1_dim, r1_dim, r2_dim, s1_dim, s2_dim, rgt_le, slv_le, cat_le):
		print('Testing on {} data'.format(dataset))

		total_count = 0
		pred_in_true = 0
		temp_within10 = 0
		temp_within20 = 0
		valid_temp_count = 0
		overall_rank1 = 0
		overall_rank3 = 0
		overall_rank15 = 0

		c1_rank_15 = 0
		s1_rank_15 = 0
		s2_rank_15 = 0
		r1_rank_15 = 0
		r2_rank_15 = 0
		s1_rank_15_on_r1_true = 0

		temp_pred_list = []
		temp_true_list = []
		for batch_num in tqdm(range(num_batches)):
			# print(batch_num)
			(x, y_true) = data_generator.next()
			rxn_ids = label_generator.next()
			rxn_ids_batch = [e for e in rxn_ids]
			# print(rxn_ids)
			# print(len(rxn_ids))
			y_pred = model.predict_on_batch(x)
			# print(y_pred.shape)
			# y_pred_bin = np.round(y_pred)
			# print(y_pred_bin[0])
			# print(sparse.csr_matrix(y_pred_bin[0]).indices)
			for i in tqdm(range(y_pred[0].shape[0])):

				# print("start iterating..")
				# print(datetime.datetime.now())
				temp_pred_avg = y_pred[5][i][0]

				true_context = []
				true_context.append(np.nonzero(y_true[0][i])[0][0])
				true_context.append(np.nonzero(y_true[1][i])[0][0])
				true_context.append(np.nonzero(y_true[2][i])[0][0])
				true_context.append(np.nonzero(y_true[3][i])[0][0])
				true_context.append(np.nonzero(y_true[4][i])[0][0])
				temp_true = y_true[5][i]

				context_combos = []
				context_combo_scores = []
				context_idvl_scores = []
				temp_preds = []
				c1_pred = np.zeros_like(y_pred[0][i],dtype = 'float32')
				c1_cdt = y_pred[0][i].argmax()
				c1_pred[c1_cdt] = 1
				
				y_pred_new = model.predict_on_batch([x[0][i].reshape(1,2048),x[1][i].reshape(1,2048),c1_pred.reshape(1,c1_dim),x[3][i].reshape(1,r1_dim),x[4][i].reshape(1,r2_dim),x[5][i].reshape(1,s1_dim),x[6][i].reshape(1,s2_dim)])
				
				s1_most_likely_five = y_pred_new[3][0].argsort()[-5:][::-1]
				for s1_cdt in s1_most_likely_five:
					s1_pred = np.zeros_like(y_pred_new[3][0])
					s1_pred[s1_cdt] = 1
					y_pred_new = model.predict_on_batch([x[0][i].reshape(1,2048),x[1][i].reshape(1,2048),c1_pred.reshape(1,c1_dim),x[3][i].reshape(1,r1_dim),x[4][i].reshape(1,r2_dim),s1_pred.reshape(1,s1_dim),x[6][i].reshape(1,s2_dim)])
				
					s2_pred = np.zeros_like(y_pred_new[4][0])
					s2_cdt = y_pred_new[4][0].argmax()
					s2_pred[s2_cdt] = 1
					y_pred_new = model.predict_on_batch([x[0][i].reshape(1,2048),x[1][i].reshape(1,2048),c1_pred.reshape(1,c1_dim),x[3][i].reshape(1,r1_dim),x[4][i].reshape(1,r2_dim),s1_pred.reshape(1,s1_dim),s2_pred.reshape(1,s2_dim)])
				
					r1_most_likely_three = y_pred_new[1][0].argsort()[-3:][::-1]



					# wait = raw_input('press any key')
			
					for r1_cdt in r1_most_likely_three:
						r1_pred = np.zeros_like(y_pred_new[1][0])
						r1_pred[r1_cdt] = 1
						y_pred_new = model.predict_on_batch([x[0][i].reshape(1,2048),x[1][i].reshape(1,2048),c1_pred.reshape(1,c1_dim),r1_pred.reshape(1,r1_dim),x[4][i].reshape(1,r2_dim),s1_pred.reshape(1,s1_dim),s2_pred.reshape(1,s2_dim)])
				
						r2_pred = np.zeros_like(y_pred_new[2][0])
						r2_cdt = y_pred_new[2][0].argmax()
						r2_pred[r2_cdt] = 1
						y_pred_new = model.predict_on_batch([x[0][i].reshape(1,2048),x[1][i].reshape(1,2048),c1_pred.reshape(1,c1_dim),r1_pred.reshape(1,r1_dim),r2_pred.reshape(1,r2_dim),s1_pred.reshape(1,s1_dim),s2_pred.reshape(1,s2_dim)])
				
						context_combo = [c1_cdt,r1_cdt,r2_cdt,s1_cdt,s2_cdt]
						context_combo_score = y_pred_new[0][0][c1_cdt]*y_pred_new[1][0][r1_cdt]*y_pred_new[2][0][r2_cdt]*y_pred_new[3][0][s1_cdt]*y_pred_new[4][0][s2_cdt]
						context_idvl_score = [y_pred_new[0][0][c1_cdt],y_pred_new[1][0][r1_cdt],y_pred_new[2][0][r2_cdt],y_pred_new[3][0][s1_cdt],y_pred_new[4][0][s2_cdt]]
						temp_pred = y_pred_new[5][0][0]

						context_combos.append(context_combo)
						context_combo_scores.append(context_combo_score)
						context_idvl_scores.append(context_idvl_score)
						temp_preds.append(temp_pred)
				
				c1_set = context_combos[0][0]
				r1_set = set([combo[1] for combo in context_combos])
				r2_set = set([combo[2] for combo in context_combos])
				s1_set = set([combo[3] for combo in context_combos])
				s2_set = set([combo[4] for combo in context_combos])

				context_ranks = 16 - stats.rankdata(context_combo_scores)
				context_combos = [context_combos[int(rank)-1] for rank in context_ranks]
				context_combo_scores = [context_combo_scores[int(rank)-1] for rank in context_ranks]
				context_idvl_scores = [context_idvl_scores[int(rank)-1] for rank in context_ranks]
				temp_preds = [temp_preds[int(rank)-1] for rank in context_ranks]
				# print("found top 3 (combo)")
				# print(datetime.datetime.now())

				true_context_score = y_pred[0][i][true_context[0]]*y_pred[1][i][true_context[1]]*y_pred[2][i][true_context[2]]*y_pred[3][i][true_context[3]]*y_pred[4][i][true_context[4]]
				if true_context in context_combos:
					true_context_rank = context_ranks[context_combos.index(true_context)]
				else:
					true_context_rank = ">15"

				
				if true_context[0] == c1_set:
					c1_rank_15 += 1
				if true_context[1] in r1_set:
					r1_rank_15+=1
				if true_context[2] in r2_set:
					r2_rank_15+=1
				if true_context[3] in s1_set:
					s1_rank_15+=1
				if true_context[4] in s2_set:
					s2_rank_15+=1
				if (true_context[1] in r1_set) and (true_context[3] in s1_set):
					s1_rank_15_on_r1_true +=1
				# print("found true context rank and score")
				# print(datetime.datetime.now())
				# print (temp_true is np.nan)
				if np.isnan(temp_true):
					temp_pred = np.nan
					# print(temp_true)
					# print(temp_pred)
				else:
					temp_pred = temp_pred_avg
					valid_temp_count += 1
					if np.abs(temp_pred - temp_true)<=10:
						temp_within10 += 1
					if np.abs(temp_pred - temp_true)<=20:
						temp_within20 += 1

				temp_true_list.append(temp_true)
				temp_pred_list.append(temp_pred)
				# true_context.append(temp_true)
				total_count+=1

				#separate and overall accuracy
				
				if true_context_rank == 1:
					overall_rank1 += 1
				if true_context_rank >= 1 and true_context_rank<=3:
					overall_rank3 +=1
				if true_context_rank >= 1 and true_context_rank<=15:
					overall_rank15 +=1

				# print(y_true[i],np.rint(y_pred[i]))
				# print corr
				# print(i)
				# true_context = sparse.csr_matrix(y_true[i]).indices,
				# pred_context = sparse.csr_matrix(y_pred_bin[i]).indices,
				#sample every 1000 reactions
				# print("calculated accuracy")
				# print(datetime.datetime.now())
				# wait = input('press something to continue...')	

				# print(true_context)
				# print(temp_true)
				# print(true_context_score)
				# print(true_context_rank)
				# print(combo_top_3)
				# print(c1_most_likely_three)
				# print(r1_most_likely_three)
				# print(r2_most_likely_three)
				# print(s1_most_likely_three)
				# print(s2_most_likely_three)
				# raw_input("press anything to continue...")

				if total_count%1 == 0 :
					# true_context_id = [0,0,0,0,0]
					# true_context_id[0] = cat_le.inverse_transform(true_context[0])
					# true_context_id[1] = rgt_le.inverse_transform(true_context[1])
					# true_context_id[2] = rgt_le.inverse_transform(true_context[2])
					# true_context_id[3] = slv_le.inverse_transform(true_context[3])
					# true_context_id[4] = slv_le.inverse_transform(true_context[4])
					# # c1_id = [cat_le.inverse_transform(c1) for c1 in c1_most_likely_three]
					# # r1_id = [rgt_le.inverse_transform(r1) for r1 in r1_most_likely_three]
					# # r2_id = [rgt_le.inverse_transform(r2) for r2 in r2_most_likely_three]
					# # s1_id = [slv_le.inverse_transform(s1) for s1 in s1_most_likely_three]
					# # s2_id = [slv_le.inverse_transform(s2) for s2 in s2_most_likely_three]
					# pred_context_1_id = [0]*5
					# pred_context_1_id[0] = cat_le.inverse_transform(combo_top_3[0][0])
					# pred_context_1_id[1] = rgt_le.inverse_transform(combo_top_3[0][1])
					# pred_context_1_id[2] = rgt_le.inverse_transform(combo_top_3[0][2])
					# pred_context_1_id[3] = slv_le.inverse_transform(combo_top_3[0][3])
					# pred_context_1_id[4] = slv_le.inverse_transform(combo_top_3[0][4])

					# pred_context_2_id = [0]*5
					# pred_context_2_id[0] = cat_le.inverse_transform(combo_top_3[1][0])
					# pred_context_2_id[1] = rgt_le.inverse_transform(combo_top_3[1][1])
					# pred_context_2_id[2] = rgt_le.inverse_transform(combo_top_3[1][2])
					# pred_context_2_id[3] = slv_le.inverse_transform(combo_top_3[1][3])
					# pred_context_2_id[4] = slv_le.inverse_transform(combo_top_3[1][4])

					# pred_context_3_id = [0]*5
					# pred_context_3_id[0] = cat_le.inverse_transform(combo_top_3[2][0])
					# pred_context_3_id[1] = rgt_le.inverse_transform(combo_top_3[2][1])
					# pred_context_3_id[2] = rgt_le.inverse_transform(combo_top_3[2][2])
					# pred_context_3_id[3] = slv_le.inverse_transform(combo_top_3[2][3])
					# pred_context_3_id[4] = slv_le.inverse_transform(combo_top_3[2][4])
					# try:
					# 	true_context_cn = [chemical_db.find_one({'_id':chem_id}, ["IDE_CN"])["IDE_CN"].replace('\n','')  if chem_id!= -1 else "None" for chem_id in true_context_id]
					# 	# c1_cn = [chemical_db.find_one({'_id':chem_id}, ["IDE_CN"])["IDE_CN"].replace('\n','') if chem_id!= -1 else "None" for chem_id in c1_id]
					# 	# r1_cn = [chemical_db.find_one({'_id':chem_id}, ["IDE_CN"])["IDE_CN"].replace('\n','')  if chem_id!= -1 else "None" for chem_id in r1_id]
					# 	# r2_cn = [chemical_db.find_one({'_id':chem_id}, ["IDE_CN"])["IDE_CN"].replace('\n','')  if chem_id!= -1 else "None" for chem_id in r2_id]
					# 	# s1_cn = [chemical_db.find_one({'_id':chem_id}, ["IDE_CN"])["IDE_CN"].replace('\n','')  if chem_id!= -1 else "None" for chem_id in s1_id]
					# 	# s2_cn = [chemical_db.find_one({'_id':chem_id}, ["IDE_CN"])["IDE_CN"].replace('\n','')  if chem_id!= -1 else "None" for chem_id in s2_id]
					# 	pred_context_1_cn = [chemical_db.find_one({'_id':chem_id}, ["IDE_CN"])["IDE_CN"].replace('\n','')  if chem_id!= -1 else "None" for chem_id in pred_context_1_id]
					# 	pred_context_2_cn = [chemical_db.find_one({'_id':chem_id}, ["IDE_CN"])["IDE_CN"].replace('\n','')  if chem_id!= -1 else "None" for chem_id in pred_context_2_id]
					# 	pred_context_3_cn = [chemical_db.find_one({'_id':chem_id}, ["IDE_CN"])["IDE_CN"].replace('\n','')  if chem_id!= -1 else "None" for chem_id in pred_context_3_id]
					# except:
					# 	print("One or more chemcial name not found, skip this entry...")
					# 	continue	

					# true_context_text = ','.join(true_context_cn)
					# pred_context_1_text = ','.join(pred_context_1_cn)
					# pred_context_2_text = ','.join(pred_context_2_cn)
					# pred_context_3_text = ','.join(pred_context_3_cn)
					true_context_text = ','.join([str(e) for e in true_context])
					# pred_context_1_text = ','.join([str(e) for e in context_combos])
					# pred_context_2_text = ','.join([str(e) for e in context_combos[1]])
					# pred_context_3_text = ','.join([str(e) for e in context_combos[2]])
					# pred_c1 = ','.join(c1_cn)
					# pred_r1 = ','.join(r1_cn)
					# pred_r2 = ','.join(r2_cn)
					# pred_s1 = ','.join(s1_cn)
					# pred_s2 = ','.join(s2_cn)

					# rxn_smiles = reaction_db.find_one({'_id':rxn_ids_batch[i]},["RXN_SMILES"])["RXN_SMILES"]
					# if (y_true[i] + np.rint(y_pred[i])>=1):
					# print("found names of chemicals")
					# print(datetime.datetime.now())
					fid.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
								dataset,
								rxn_ids_batch[i],
								# rxn_smiles,
								true_context_text,
								true_context_score,
								true_context_rank,
								context_combos,
								context_combo_scores,
								context_idvl_scores,
								temp_preds,
								temp_true,
								temp_pred
							))
					# print("wrote to files...")
					# print(datetime.datetime.now())
					# wait = input('press something to continue...')

		# print(corr, len(outcome_true))
		overall_top1_acc = overall_rank1/total_count
		overall_top3_acc = overall_rank3/total_count
		overall_top15_acc = overall_rank15/total_count

		c1_top15_acc = c1_rank_15/total_count
		r1_top15_acc = r1_rank_15/total_count
		r2_top15_acc = r2_rank_15/total_count
		s1_top15_acc = s1_rank_15/total_count
		s2_top15_acc = s2_rank_15/total_count
		s1_top15_and_r1_top15 = s1_rank_15_on_r1_true/total_count
		s1_top3_on_r1_top5 = s1_rank_15_on_r1_true/s1_rank_15

		temp_within10_pct = temp_within10/valid_temp_count
		temp_within20_pct = temp_within20/valid_temp_count
		
		# mse_temp = ((np.array(temp_pred_list)-np.array(temp_true_list))**2).mean()
		fid.write('overall: {}\t overall_top1: {}\t overall_top3: {}\t overall_top15: {}\t c1_top15: {}\t r1_top15: {}\t r2_top15: {}\t s1_top15: {}\t s2_top15: {}\t s1_top15_and_r1_top_15: {}\t s1_top15_on_r1_top_15: {}\t temp+-10: {}\t temp+-20: {}\t\n\n\n\n'.format(
					dataset, 
					overall_top1_acc,
					overall_top3_acc,
					overall_top15_acc,
					c1_top15_acc,
					r1_top15_acc,
					r2_top15_acc,
					s1_top15_acc,
					s2_top15_acc,
					s1_top15_and_r1_top15,
					s1_top3_on_r1_top5,
					temp_within10_pct,
					temp_within20_pct
				))
		print('overall: {}\t overall_top1: {}\t overall_top3: {}\t overall_top15: {}\t c1_top15: {}\t r1_top15: {}\t r2_top15: {}\t s1_top15: {}\t s2_top15: {}\t s1_top15_and_r1_top_15: {}\t s1_top15_on_r1_top_15: {}\t temp+-10: {}\t temp+-20: {}\t\n\n\n\n'.format(
					dataset, 
					overall_top1_acc,
					overall_top3_acc,
					overall_top15_acc,
					c1_top15_acc,
					r1_top15_acc,
					r2_top15_acc,
					s1_top15_acc,
					s2_top15_acc,
					s1_top15_and_r1_top15,
					s1_top3_on_r1_top5,
					temp_within10_pct,
					temp_within20_pct
				))
		
		return 0

	train_accu = test_on_set(fid, 'train', data['train_generator'], data['train_label_generator'], 
		int(np.ceil(data['train_nb_samples']/data['batch_size'])), c1_dim, r1_dim, r2_dim, s1_dim, s2_dim, rgt_le, slv_le, cat_le
	)
	val_accu = test_on_set(fid, 'val', data['val_generator'], data['val_label_generator'], 
		int(np.ceil(data['val_nb_samples']/data['batch_size'])), c1_dim, r1_dim, r2_dim, s1_dim, s2_dim, rgt_le, slv_le, cat_le
	)
	test_accu = test_on_set(fid, 'test', data['test_generator'], data['test_label_generator'], 
		int(np.ceil(data['test_nb_samples']/data['batch_size'])), c1_dim, r1_dim, r2_dim, s1_dim, s2_dim, rgt_le, slv_le, cat_le
	)
	# train_accu = test_on_set(fid, 'train', data['train_generator'], data['train_label_generator'], 
	# 	2, c1_dim, r1_dim, r2_dim, s1_dim, s2_dim, rgt_le, slv_le, cat_le
	# )
	# val_accu = test_on_set(fid, 'val', data['val_generator'], data['val_label_generator'], 
	# 	2, c1_dim, r1_dim, r2_dim, s1_dim, s2_dim, rgt_le, slv_le, cat_le
	# )
	# test_accu = test_on_set(fid, 'test', data['test_generator'], data['test_label_generator'], 
	# 	2, c1_dim, r1_dim, r2_dim, s1_dim, s2_dim, rgt_le, slv_le, cat_le
	# )


	fid.close()


if __name__ == '__main__':
	set_keras_backend("theano")
	rxn_id_file = "/home/hanyug/Make-It/makeit/context_pred/preprocessed_data/separate/full3/rxn_ids.pickle"
	# edit_vec_file = "/home/hanyug/Make-It/makeit/context_pred/preprocessed_data/edit_vecs.pickle"
	# context_mtx_file = "/home/hanyug/Make-It/makeit/context_pred/preprocessed_data/separate/1Msubset/context_mtx.npz"
	# context_file = "/home/hanyug/Make-It/makeit/context_pred/preprocessed_data/1Msubset/context.pickle"
	pfp_mtx_file = "/home/hanyug/Make-It/makeit/context_pred/preprocessed_data/separate/full3/pfp_mtx.npz"
	rfp_mtx_file = "/home/hanyug/Make-It/makeit/context_pred/preprocessed_data/separate/full3/rfp_mtx.npz"
	rgt_1_mtx_file = "/home/hanyug/Make-It/makeit/context_pred/preprocessed_data/separate/full3/rgt_1_mtx.npz"
	rgt_2_mtx_file = "/home/hanyug/Make-It/makeit/context_pred/preprocessed_data/separate/full3/rgt_2_mtx.npz"
	slv_1_mtx_file = "/home/hanyug/Make-It/makeit/context_pred/preprocessed_data/separate/full3/slv_1_mtx.npz"
	slv_2_mtx_file = "/home/hanyug/Make-It/makeit/context_pred/preprocessed_data/separate/full3/slv_2_mtx.npz"
	cat_1_mtx_file = "/home/hanyug/Make-It/makeit/context_pred/preprocessed_data/separate/full3/cat_1_mtx.npz"
	rgt_file = "/home/hanyug/Make-It/makeit/context_pred/preprocessed_data/separate/full3/rgts.pickle"
	slv_file = "/home/hanyug/Make-It/makeit/context_pred/preprocessed_data/separate/full3/slvs.pickle"
	cat_file = "/home/hanyug/Make-It/makeit/context_pred/preprocessed_data/separate/full3/cats.pickle"
	temp_file = "/home/hanyug/Make-It/makeit/context_pred/preprocessed_data/separate/full3/temps.pickle"
	yd_file = "/home/hanyug/Make-It/makeit/context_pred/preprocessed_data/separate/full3/yds.pickle"

	pfp_csrmtx = sparse.load_npz(pfp_mtx_file)
	rfp_csrmtx = sparse.load_npz(rfp_mtx_file)
	# context_csrmtx = sparse.load_npz(context_mtx_file)
	rgt_1_mtx = sparse.load_npz(rgt_1_mtx_file)
	rgt_2_mtx = sparse.load_npz(rgt_2_mtx_file)
	slv_1_mtx = sparse.load_npz(slv_1_mtx_file)
	slv_2_mtx = sparse.load_npz(slv_2_mtx_file)
	cat_1_mtx = sparse.load_npz(cat_1_mtx_file)
	with open(rgt_file,"r") as RGT_L_F:
		rgt_counter = pickle.load(RGT_L_F)
	with open(slv_file,"r") as SLV_L_F:
		slv_counter = pickle.load(SLV_L_F)
	with open(cat_file,"r") as CAT_L_F:
		cat_counter = pickle.load(CAT_L_F)
	with open(temp_file,"r") as T_L_F:
		temp_list = pickle.load(T_L_F)
	# cat_le = LabelEncoder()
	# slv_le = LabelEncoder()
	# rgt_le = LabelEncoder()

	# rgt_le.fit(flat_rgt_list)
	# slv_le.fit(flat_slv_list)
	# cat_le.fit(flat_cat_list)

	#log modulus transform of temp data
	# temp_list = [np.sign(x)*(np.log10(np.abs(x)+1)) if x!= -1 else np.nan for x in temp_list]

	temp_list = [x if x!= -1 else np.nan for x in temp_list]
	
	# print(temp_list[0:100])
	# wait = input("PRESS ENTER TO CONTINUE.")

	print(pfp_csrmtx.shape, rfp_csrmtx.shape)
	with open(rxn_id_file,"r") as RID:
		rxn_id_list = pickle.load(RID)
	print(len(rxn_id_list))
	# with open(edit_vec_file,"w") as EDT:
	# 	pickle.dump(edit_vec_list,EDT)
	# with open(context_file,"r") as CON_L_F:
	# 	context_list = pickle.load(CON_L_F)
	# le = LabelEncoder()
	# ohe = OneHotEncoder()
	# le.fit(context_list)

	c1_dim = cat_1_mtx.shape[1]
	r1_dim = rgt_1_mtx.shape[1]
	r2_dim = rgt_2_mtx.shape[1]
	s1_dim = slv_1_mtx.shape[1]
	s2_dim = slv_2_mtx.shape[1]
	print(c1_dim, r1_dim, r2_dim, s1_dim, s2_dim)


	parser = argparse.ArgumentParser()
	parser.add_argument('--nb_epoch', type = int, default = 100,
						help = 'Number of epochs to train for, default 100')
	parser.add_argument('--batch_size', type = int, default = 256,
						help = 'Batch size, default 256')
	parser.add_argument('--retrain', type = bool, default = False,
		                help = 'Retrain with loaded weights, default False')
	parser.add_argument('--test', type = bool, default = False,
						help = 'Test model only, default False')
	parser.add_argument('--l2', type = float, default = 0,
						help = 'l2 regularization parameter for each Dense layer, default 0')
	parser.add_argument('--lr', type = float, default = 0.001, 
						help = 'Learning rate, default 0.001')
	args = parser.parse_args()

	# mol = Chem.MolFromSmiles('[C:1][C:2]')
	# (a, _, b, _) = edits_to_vectors((['1'],[],[('1','2',1.0)],[]), mol)

	# F_atom = len(a[0])
	# F_bond = len(b[0])
	
	nb_epoch           = int(args.nb_epoch)
	batch_size         = int(args.batch_size)
	l2v                = float(args.l2)
	lr 				   = float(args.lr)

	
	MODEL_FPATH = "/home/hanyug/Make-It/makeit/context_pred/model/rgt_to_cat/model.json"
	WEIGHTS_FPATH = "/home/hanyug/Make-It/makeit/context_pred/model/rgt_to_cat/weights.h5"
	BEST_WEIGHTS_FPATH = "/home/hanyug/Make-It/makeit/context_pred/model/rgt_to_cat/best_weights.h5"
	LOG_FPATH = "/home/hanyug/Make-It/makeit/context_pred/model/rgt_to_cat/training.log"
	# if bool(args.retrain):
	# 	print('Reloading from file')
	# 	rebuild = raw_input('Do you want to rebuild from scratch instead of loading from file? [n/y] ')
	# 	if rebuild == 'y':
	# 		model = build(pfp_len = 2048, rxnfp_len = 2048, output_dim = output_dim, N_h1 = 4096, N_h2 = 0)
	# 	else:
	# 		model = model_from_json(open(MODEL_FPATH).read())
	# 		model.compile(loss = 'categorical_crossentropy', 
	# 			optimizer = opt,
	# 			metrics = ['accuracy'])
	# 	model.load_weights(WEIGHTS_FPATH)
	# else:
	# 	model = build(pfp_len = 2048, rxnfp_len = 2048, output_dim = output_dim, N_h1 = 4096, N_h2 = 0)
	# 	try:
	# 		with open(MODEL_FPATH, 'w') as outfile:
	# 			outfile.write(model.to_json())
	# 	except:
	# 		print('could not write model to json')

	# if bool(args.test):
	# 	test(model, h5f, SPLIT_RATIO)
	# 	quit(1)
	split_ratio = (0.8,0.1)
	class_weight = None
	model = build(pfp_len = 16384, rxnfp_len = 16384, c1_dim = c1_dim, r1_dim = r1_dim, r2_dim = r2_dim, s1_dim = s1_dim, s2_dim = s2_dim, N_h1 = 300, N_h2 = 100, l2v = l2v, lr = lr)

	with open(MODEL_FPATH, 'w') as outfile:
		outfile.write(model.to_json())

		print('write model to json')


	train(model, pfp_csrmtx, rfp_csrmtx, cat_1_mtx, rgt_1_mtx, rgt_2_mtx, slv_1_mtx, slv_2_mtx, temp_list, split_ratio, class_weight,batch_size)
	model.save_weights(WEIGHTS_FPATH, overwrite = True) 

	###load model##############
	# load json and create model
	# json_file = open(MODEL_FPATH, 'r')
	# loaded_model_json = json_file.read()
	# json_file.close()
	# model = model_from_json(loaded_model_json)
	
	# load weights into new model
	# model.load_weights(WEIGHTS_FPATH)

	# test(model, pfp_csrmtx, rfp_csrmtx, cat_1_mtx, rgt_1_mtx, rgt_2_mtx, slv_1_mtx, slv_2_mtx, temp_list, split_ratio, class_weight, c1_dim, r1_dim, r2_dim, s1_dim, s2_dim, rgt_le, slv_le, cat_le)

	# h5f.close()