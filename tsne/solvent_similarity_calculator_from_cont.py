import numpy as np
import pickle
from keras.models import model_from_json
from keras import backend as K
from scipy.spatial.distance import pdist
import pandas as pd

class solvent_similarity_calculator():
	def __init__(self):
		self.model = None
		self.s1_dict = None


	def load(self, model_path='', dict_path='', weights_path=''):
		s1_dict_file = dict_path + "s1_dict.pickle"
		with open(s1_dict_file, "r") as S1_DICT_F:
			self.s1_dict = pickle.load(S1_DICT_F)

		json_file = open(model_path, 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		self.model = model_from_json(loaded_model_json)
		# load weights into new model
		self.model.load_weights(weights_path)
		self.s1_dim = self.model.input_shape[5][1]
		slv_input_layer = self.model.get_layer('s1')
		self.weights = slv_input_layer.get_weights()
		print(self.weights[1].shape)
		
	def smiles_to_cat(self, smiles = ''):
		cat = -1
		for s_cat, s_smiles in self.s1_dict.iteritems():
			if s_smiles == smiles:
				cat = s_cat
		if cat ==-1:
			print('smiles not found!{}'.format(smiles))

		return cat
	def get_slv_fp(self, slv, exponent=False):
		# print(np.transpose(self.weights).shape)
		cat = self.smiles_to_cat(slv)
		if cat==-1:
			return None
		slv_fp = np.transpose(self.weights[0])[cat,:]
		slv_fp_bias = np.transpose(self.weights[1])[cat]
		return slv_fp

	def solv_sim(self, slv_1, slv_2):
		# calculate the similarity of two solven represented with smiles (or name if smiles not available)
		# cat_1 = self.smiles_to_cat(slv_1)
		# cat_2 = self.smiles_to_cat(slv_2)
		# slv_1_onehot = np.zeros([1,self.s1_dim])
		# slv_2_onehot = np.zeros([1,self.s1_dim])
		# slv_1_onehot[0,cat_1-1] = 1
		# slv_2_onehot[0,cat_2-1] = 1
		
		slv_1_fp = self.get_slv_fp(slv_1)
		slv_2_fp = self.get_slv_fp(slv_2)
		# print(slv_1_fp)
		# print(slv_2_fp)
		similarity = np.inner(slv_1_fp,slv_2_fp)/np.sqrt(np.inner(slv_1_fp,slv_1_fp)*np.inner(slv_2_fp,slv_2_fp))
		return similarity

