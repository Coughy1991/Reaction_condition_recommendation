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
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Fingerprints import FingerprintMols
import numpy as np
import pickle
from pymongo import MongoClient
from scipy import sparse
from tqdm import tqdm
from neuralnetwork import NeuralNetContextRecommender
from train_model_c_s_r_deploy import batch_data_generator,batch_label_generator,load_and_partition_data
import datetime
import pandas as pd
from sklearn.metrics import mean_squared_error as mse
# from solvent_similarity_calculator import SolventSimCalc, is_similar_reagent
import random
# def is_similar_reagent(rgt1, rgt2):
# 	list_of_metal_atoms = ['Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag',\
# 						   'Cd','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Rf','Db','Sg','Bh','Hs','Mt','Ds','Rg','Cn',\
# 						   'La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu',\
# 						   'Ac','Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm','Md','No','Lr'
# 						   ]
# 	if rgt1 == rgt2:
# 		return True
# 	elif 'Reaxys ID' in rgt1 or 'Reaxys ID' in rgt2:
# 		return False	
# 	else:
# 		#if have metal atoms compare the 
# 		if any(metal in rgt1 for metal in list_of_metal_atoms):
# 			rgt1 = [metal for metal in list_of_metal_atoms if metal in rgt1]

# 		if any(metal in rgt2 for metal in list_of_metal_atoms):
# 			rgt2 = [metal for metal in list_of_metal_atoms if metal in rgt2]

# 		if rgt1 == rgt2:
# 			return True

# 		if 'Reaxys' in rgt1 or 'Reaxys' in rgt2:
# 			return False
# 		try:
# 			fp1 = FingerprintMols.FingerprintMol(rgt1)
# 			fp2 = FingerprintMols.FingerprintMol(rgt2)
# 		except:
# 			return False
# 		similarity = DataStructs.FingerprintSimilarity(fp1,fp2)
# 		if similarity >=1.0:
# 			return True
# 		else:
# 			return False

list_of_metal_atoms = ['Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag',\
							'Cd','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Rf','Db','Sg','Bh','Hs','Mt','Ds','Rg','Cn',
							'Ln',
							'Ce',
							'Pr',
							'Nd',
							'Pm',
							'Sm',
							'Eu',
							'Gd',
							'Tb',
							'Dy',
							'Ho',
							'Er',
							'Tm',
							'Yb',
							'Lu',
							'Ac',
							'Th',
							'Pa',
							'U',
							'Np',
							'Am',
							'Cm',
							'Bk',
							'Cf',
							'Es',
							'Fm',
							'Md',
							'No',
							'Lr',
							]
list_of_full_metal_names = ['Scandium',
							'Titanium',
							'Vanadium',
							'Chromium',
							'Manganese',
							'Iron',
							'Cobalt',
							'Nickel',
							'Copper',
							'Zinc',
							'Yttrium',
							'Zirconium',
							'Niobium',
							'Molybdenum',
							'Technetium',
							'Ruthenium',
							'Rhodium',
							'Palladium',
							'Silver',
							'Cadmium',
							'Hafnium',
							'Tantalum',
							'Tungsten',
							'Rhenium',
							'Osmium',
							'Iridium',
							'Platinum',
							'Gold',
							'Mercury',
							'Rutherfordium',
							'Dubnium',
							'Seaborgium',
							'Bohrium',
							'Hassium',
							'Meitnerium',
							'Darmstadtium',
							'Roentgenium',
							'Copernicium',
							'Lanthanum',
							'Cerium',
							'Praseodymium',
							'Neodymium',
							'Promethium',
							'Samarium',
							'Europium',
							'Gadolinium',
							'Terbium',
							'Dysprosium',
							'Holmium',
							'Erbium',
							'Thulium',
							'Ytterbium',
							'Lutetium',
							'Actinium',
							'Thorium',
							'Protactinium',
							'Uranium',
							'Neptunium',
							'Plutonium',
							'Americium',
							'Curium',
							'Berkelium',
							'Californium',
							'Einsteinium',
							'Fermium',
							'Mendelevium',
							'Nobelium',
							'Lawrencium',
							]
def is_similar_reagent(rgt1, rgt2,list_of_metal_atoms,list_of_full_metal_names):
	if rgt1 == rgt2:
		return True
	elif 'Reaxys ID' in rgt1 or 'Reaxys ID' in rgt2:
		return False	
	else:
		#if have metal atoms compare the 
		rgt1_metal = 100
		rgt2_metal = 101
		if any(metal in rgt1 for metal in list_of_full_metal_names):
			rgt1_metal = [list_of_full_metal_names.index(metal) for metal in list_of_full_metal_names if metal in rgt1]
		elif any(metal in rgt1 for metal in list_of_metal_atoms):
			rgt1_metal = [list_of_metal_atoms.index(metal) for metal in list_of_metal_atoms if metal in rgt1]
		
		if any(metal in rgt2 for metal in list_of_full_metal_names):
			rgt2_metal = [list_of_full_metal_names.index(metal) for metal in list_of_full_metal_names if metal in rgt2]
		elif any(metal in rgt2 for metal in list_of_metal_atoms):
			rgt2_metal = [list_of_metal_atoms.index(metal) for metal in list_of_metal_atoms if metal in rgt2]

		if rgt1_metal == rgt2_metal:
			return True

		if 'Reaxys' in rgt1 or 'Reaxys' in rgt2:
			return False
		try:
			mol1 = Chem.MolFromSmiles(rgt1)
			mol2 = Chem.MolFromSmiles(rgt2)
			fp1 = FingerprintMols.FingerprintMol(mol1)
			fp2 = FingerprintMols.FingerprintMol(mol2)
		except:
			print('cannot calculate fp')
			return False
		if not any(list(fp1)) or not any(list(fp2)):
			return False
		similarity = DataStructs.FingerprintSimilarity(fp1,fp2)
		if similarity >=1.0:
			return True
		else:
			return False
# print(is_similar_reagent('[Na+].[OH-]','[Li+].[OH-]',list_of_metal_atoms,list_of_full_metal_names))
# print(is_similar_reagent('Palladium','Pd',list_of_metal_atoms,list_of_full_metal_names))
# print(is_similar_reagent('Cl','Br',list_of_metal_atoms,list_of_full_metal_names))



# client = ###

# db = client['prediction']
# reaxys_db = client['reaxys_v2']
# reaction_db = reaxys_db['reactions']

# db_client = ###
# sdb = db_client[gc.SOLVENTS['database']]
# SOLVENT_DB = sdb[gc.SOLVENTS['collection']]
# print(SOLVENT_DB.find_one({}))
##load context recommender
cont = NeuralNetContextRecommender()
    # cont.load_nn_model(model_path = "/home/hanyug/Make-It/makeit/context_pred/model/c_s_r_fullset/model.json", info_path = "/home/hanyug/Make-It/makeit/context_pred/preprocessed_data/separate/fullset2048/", weights_path="/home/hanyug/Make-It/makeit/context_pred/model/c_s_r_fullset/weights.h5")
cont.load_nn_model(model_path="/home/hanyug/Make-It/makeit/data/context/NeuralNet_Cont_Model/model.json", 
        info_path="/home/hanyug/Make-It/makeit/data/context/NeuralNet_Cont_Model/", 
        weights_path="/home/hanyug/Make-It/makeit/data/context/NeuralNet_Cont_Model/weights.h5")
# cont.load_nn_model(model_path="/home/hanyug/Make-It/makeit/context_pred/model/test/model.json", info_path=gc.NEURALNET_CONTEXT_REC[
#                    'info_path'], weights_path="/home/hanyug/Make-It/makeit/context_pred/model/test/weights.h5")


# simcal = SolventSimCalc()
# simcal.load_params(SOLVENT_DB)

rxn_id_file = "/data/hanyu/preprocessed_data/separate/fullset_rearrange_test_chiral/rxn_ids.pickle"
pfp_mtx_file = "/data/hanyu/preprocessed_data/separate/fullset_rearrange_test_chiral/pfp_mtx.npz"
rfp_mtx_file = "/data/hanyu/preprocessed_data/separate/fullset_rearrange_test_chiral/rfp_mtx.npz"
rgt_1_mtx_file = "/data/hanyu/preprocessed_data/separate/fullset_rearrange_test_chiral/rgt_1_mtx.npz"
rgt_2_mtx_file = "/data/hanyu/preprocessed_data/separate/fullset_rearrange_test_chiral/rgt_2_mtx.npz"
slv_1_mtx_file = "/data/hanyu/preprocessed_data/separate/fullset_rearrange_test_chiral/slv_1_mtx.npz"
slv_2_mtx_file = "/data/hanyu/preprocessed_data/separate/fullset_rearrange_test_chiral/slv_2_mtx.npz"
cat_1_mtx_file = "/data/hanyu/preprocessed_data/separate/fullset_rearrange_test_chiral/cat_1_mtx.npz"
rgt_file = "/data/hanyu/preprocessed_data/separate/fullset_rearrange_test_chiral/rgts.pickle"
slv_file = "/data/hanyu/preprocessed_data/separate/fullset_rearrange_test_chiral/slvs.pickle"
cat_file = "/data/hanyu/preprocessed_data/separate/fullset_rearrange_test_chiral/cats.pickle"
temp_file = "/data/hanyu/preprocessed_data/separate/fullset_rearrange_test_chiral/temps.pickle"
yd_file = "/data/hanyu/preprocessed_data/separate/fullset_rearrange_test_chiral/yds.pickle"

pfp_csrmtx = sparse.load_npz(pfp_mtx_file)
rfp_csrmtx = sparse.load_npz(rfp_mtx_file)
# context_csrmtx = sparse.load_npz(context_mtx_file)
rgt_1_mtx = sparse.load_npz(rgt_1_mtx_file)
rgt_2_mtx = sparse.load_npz(rgt_2_mtx_file)
slv_1_mtx = sparse.load_npz(slv_1_mtx_file)
slv_2_mtx = sparse.load_npz(slv_2_mtx_file)
cat_1_mtx = sparse.load_npz(cat_1_mtx_file)

with open(temp_file,"r") as T_L_F:
	temp_list = pickle.load(T_L_F)

with open(rxn_id_file,"r") as RID:
	rxn_id_list = pickle.load(RID)	


# print(rxn_smiles_list)
split_ratio = [0.8,0.1]

data = load_and_partition_data(pfp_csrmtx, rfp_csrmtx, rxn_id_list, cat_1_mtx, rgt_1_mtx, rgt_2_mtx, slv_1_mtx, slv_2_mtx, temp_list, split_ratio, batch_size = 1)

outfile = open('/home/hanyug/Make-It/makeit/context_pred/results_analysis/predict_results.dat','w')
summary_file = open('/home/hanyug/Make-It/makeit/context_pred/results_analysis/summary.dat','w')

ctr = 0
top3_ctr = 0
top10_ctr = 0
top3_similar_ctr=0
top10_similar_ctr=0
c1s1r1_top3 = 0
c1s1r1_top10 = 0
c1s1r1_similar_top3 = 0
c1s1r1_similar_top10 = 0
s1_top3_ctr = 0
s1_top10_ctr = 0
s1_top3_similar_ctr = 0
s1_top10_similar_ctr = 0
r1_top3_ctr = 0
r1_top10_ctr = 0
r1_top3_similar_ctr = 0
r1_top10_similar_ctr = 0
s2_top3_ctr = 0
s2_top10_ctr = 0
s2_top3_similar_ctr = 0
s2_top10_similar_ctr = 0
r2_top3_ctr = 0
r2_top10_ctr = 0
r2_top3_similar_ctr = 0
r2_top10_similar_ctr = 0
c1_top3_ctr = 0
c1_top10_ctr = 0
c1_top3_similar_ctr = 0
c1_top10_similar_ctr = 0

top3_ctr_null = 0
top10_ctr_null = 0
top3_similar_ctr_null=0
top10_similar_ctr_null=0
c1s1r1_top3 = 0
c1s1r1_top10 = 0
c1s1r1_similar_top3 = 0
c1s1r1_similar_top10 = 0
s1_top3_ctr_null = 0
s1_top10_ctr_null = 0
s1_top3_similar_ctr_null = 0
s1_top10_similar_ctr_null = 0
r1_top3_ctr_null = 0
r1_top10_ctr_null = 0
r1_top3_similar_ctr_null = 0
r1_top10_similar_ctr_null = 0
s2_top3_ctr_null = 0
s2_top10_ctr_null = 0
s2_top3_similar_ctr_null = 0
s2_top10_similar_ctr_null = 0
r2_top3_ctr_null = 0
r2_top10_ctr_null = 0
r2_top3_similar_ctr_null = 0
r2_top10_similar_ctr_null = 0
c1_top3_ctr_null = 0
c1_top10_ctr_null = 0
c1_top3_similar_ctr_null = 0
c1_top10_similar_ctr_null = 0

rid_list = []
rxn_smiles_list = []
true_context_list = []
top_pred_list = []
closest_pred_list = []
true_context_rank_list = []
temp_true_list = []
temp_pred_list = []
closest_temp_pred = []
top3_similar_list = []
top10_similar_list = []
c1s1r1_top3_list = []
c1s1r1_top10_list = []
c1s1r1_similar_top3_list = []
c1s1r1_similar_top10_list = []
s1_top3_list = []
s1_top10_list = []
s1_top3_similar_list = []
s1_top10_similar_list = []
s2_top3_list = []
s2_top10_list = []
s2_top3_similar_list = []
s2_top10_similar_list = []
r1_top3_list = []
r1_top10_list = []
r1_top3_similar_list = []
r1_top10_similar_list = []
r2_top3_list = []
r2_top10_list = []
r2_top3_similar_list = []
r2_top10_similar_list = []
c1_top3_list = []
c1_top10_list = []
c1_top3_similar_list = []
c1_top10_similar_list = []

true_context_rank_list_null = []
top3_similar_list_null = []
top10_similar_list_null = []
c1s1r1_top3_list_null = []
c1s1r1_top10_list_null = []
c1s1r1_similar_top3_list_null = []
c1s1r1_similar_top10_list_null = []
s1_top3_list_null = []
s1_top10_list_null = []
s1_top3_similar_list_null = []
s1_top10_similar_list_null = []
s2_top3_list_null = []
s2_top10_list_null = []
s2_top3_similar_list_null = []
s2_top10_similar_list_null = []
r1_top3_list_null = []
r1_top10_list_null = []
r1_top3_similar_list_null = []
r1_top10_similar_list_null = []
r2_top3_list_null = []
r2_top10_list_null = []
r2_top3_similar_list_null = []
r2_top10_similar_list_null = []
c1_top3_list_null = []
c1_top10_list_null = []
c1_top3_similar_list_null = []
c1_top10_similar_list_null = []

null_model_combos = [['','','','',''],
					 ['','ClCCl','','',''],
					 ['','C1CCOC1','','',''],
					 ['','','','CCN(CC)CC',''],
					 ['','','','O=C([O-])[O-].[K+]',''],
					 ['','ClCCl','','CCN(CC)CC',''],
					 ['','C1CCOC1','','CCN(CC)CC',''],
					 ['','ClCCl','','O=C([O-])[O-].[K+]',''],
					 ['','C1CCOC1','','O=C([O-])[O-].[K+]',''],
					 ['Reaxys Name palladium on activated charcoal','','','','']
					]


pct = 0.01
# data['test_nb_samples']
for i in tqdm(range(100)):
	#generate a random number and decide whether or not to skip this item
	# rn = random.uniform(0,1)
	(x, y_true) = data['test_generator'].next()
	rxn_ids = data['test_label_generator'].next()
	# if rn>pct: continue
	# rxn_smiles = ""
	try:
		rxn_smiles = reaction_db.find_one({'_id': rxn_ids[0]},['RXN_SMILES'])['RXN_SMILES']
	except:
		print('smiles not found for the reaction!{}'.format(rxn_ids[0]))
		pass
	pfp = x[0]
	rxnfp = x[1]
	c1_input = []
	r1_input = []
	r2_input = []
	s1_input = []
	s2_input = []
	true_context = []
	true_context.append(np.nonzero(y_true[0][0])[0][0])
	true_context.append(np.nonzero(y_true[3][0])[0][0])
	true_context.append(np.nonzero(y_true[4][0])[0][0])
	true_context.append(np.nonzero(y_true[1][0])[0][0])
	true_context.append(np.nonzero(y_true[2][0])[0][0])
	true_context_smiles = [0]*5
	true_context_smiles[0] = cont.category_to_name('c1',true_context[0])
	true_context_smiles[1] = cont.category_to_name('s1',true_context[1])
	true_context_smiles[2] = cont.category_to_name('s2',true_context[2])
	true_context_smiles[3] = cont.category_to_name('r1',true_context[3])
	true_context_smiles[4] = cont.category_to_name('r2',true_context[4])

	temp_true = y_true[5][0]
	inputs = [pfp,rxnfp,c1_input,r1_input,r2_input,s1_input,s2_input]
	(context_combos,context_combo_scores) = cont.predict_top_combos(inputs,return_categories_only=True, c1_rank_thres=2, r1_rank_thres=3)
	pred_context_smiles_list = []
	top3_flag = 0
	top10_flag = 0
	top3_similar_flag = 0
	top10_similar_flag = 0
	c1s1r1_top3_flag = 0
	c1s1r1_top10_flag = 0
	c1s1r1_similar_top3_flag = 0
	c1s1r1_similar_top10_flag = 0
	s1_top3_flag = 0
	s1_top10_flag= 0
	s1_top3_similar_flag = 0
	s1_top10_similar_flag= 0
	r1_top3_flag = 0
	r1_top10_flag = 0
	r1_top3_similar_flag = 0
	r1_top10_similar_flag = 0
	s2_top3_flag = 0
	s2_top10_flag= 0
	s2_top3_similar_flag = 0
	s2_top10_similar_flag= 0
	r2_top3_flag = 0
	r2_top10_flag = 0
	r2_top3_similar_flag = 0
	r2_top10_similar_flag = 0
	c1_top3_flag = 0
	c1_top10_flag=0
	c1_top3_similar_flag = 0
	c1_top10_similar_flag = 0

	######
	top3_flag_null = 0
	top10_flag_null = 0
	top3_similar_flag_null = 0
	top10_similar_flag_null = 0
	c1s1r1_top3_flag_null = 0
	c1s1r1_top10_flag_null = 0
	c1s1r1_similar_top3_flag_null = 0
	c1s1r1_similar_top10_flag_null = 0
	s1_top3_flag_null = 0
	s1_top10_flag_null= 0
	s1_top3_similar_flag_null = 0
	s1_top10_similar_flag_null= 0
	r1_top3_flag_null = 0
	r1_top10_flag_null = 0
	r1_top3_similar_flag_null = 0
	r1_top10_similar_flag_null = 0
	s2_top3_flag_null = 0
	s2_top10_flag_null= 0
	s2_top3_similar_flag_null = 0
	s2_top10_similar_flag_null= 0
	r2_top3_flag_null = 0
	r2_top10_flag_null = 0
	r2_top3_similar_flag_null = 0
	r2_top10_similar_flag_null = 0
	c1_top3_flag_null = 0
	c1_top10_flag_null=0
	c1_top3_similar_flag_null = 0
	c1_top10_similar_flag_null = 0

	true_context_rank = 100
	true_context_rank_null = 100
	for j in range(10):
		pred_context_smiles = [0]*5
		# print context_combos[j][0]
		pred_context_smiles[0] = cont.category_to_name('c1',context_combos[j][0])
		pred_context_smiles[1] = cont.category_to_name('s1',context_combos[j][1])
		pred_context_smiles[2] = cont.category_to_name('s2',context_combos[j][2])
		pred_context_smiles[3] = cont.category_to_name('r1',context_combos[j][3])
		pred_context_smiles[4] = cont.category_to_name('r2',context_combos[j][4])
		pred_true_match = [0]*5
		pred_true_match[0] = (context_combos[j][0] == true_context[0])
		pred_true_match[1] = (context_combos[j][1] == true_context[1])
		pred_true_match[2] = (context_combos[j][2] == true_context[2])
		pred_true_match[3] = (context_combos[j][3] == true_context[3])
		pred_true_match[4] = (context_combos[j][4] == true_context[4])
		pred_true_similar = [0]*5
		pred_true_similar[0] = is_similar_reagent(true_context_smiles[0],pred_context_smiles[0],list_of_metal_atoms,list_of_full_metal_names)
		pred_true_similar[1] = 1
		pred_true_similar[2] = 1
		# pred_true_similar[1] = simcal.is_similar_solvent(true_context_smiles[1],pred_context_smiles[1],0.303)
		# pred_true_similar[2] = simcal.is_similar_solvent(true_context_smiles[2],pred_context_smiles[2],0.303)
		pred_true_similar[3] = is_similar_reagent(true_context_smiles[3],pred_context_smiles[3],list_of_metal_atoms,list_of_full_metal_names)
		pred_true_similar[4] = is_similar_reagent(true_context_smiles[4],pred_context_smiles[4],list_of_metal_atoms,list_of_full_metal_names)


		####null model results
		# true_context_smiles[0] = cont.category_to_name('c1',true_context[0])
		# true_context_smiles[1] = cont.category_to_name('s1',true_context[1])
		# true_context_smiles[2] = cont.category_to_name('s2',true_context[2])
		# true_context_smiles[3] = cont.category_to_name('r1',true_context[3])
		# true_context_smiles[4] = cont.category_to_name('r2',true_context[4])
		
		null_true_match = [0]*5
		null_true_match[0] = (null_model_combos[j][0] == true_context_smiles[0])
		null_true_match[1] = (null_model_combos[j][1] == true_context_smiles[1])
		null_true_match[2] = (null_model_combos[j][2] == true_context_smiles[2])
		null_true_match[3] = (null_model_combos[j][3] == true_context_smiles[3])
		null_true_match[4] = (null_model_combos[j][4] == true_context_smiles[4])
		null_true_similar = [0]*5
		null_true_similar[0] = is_similar_reagent(true_context_smiles[0],null_model_combos[j][0],list_of_metal_atoms,list_of_full_metal_names)
		null_true_similar[1] = 1
		null_true_similar[2] = 1
		# null_true_similar[1] = simcal.is_similar_solvent(true_context_smiles[1],null_model_combos[j][1],0.303)
		# null_true_similar[2] = simcal.is_similar_solvent(true_context_smiles[2],null_model_combos[j][2],0.303)

		null_true_similar[3] = is_similar_reagent(true_context_smiles[3],null_model_combos[j][3],list_of_metal_atoms,list_of_full_metal_names)
		null_true_similar[4] = is_similar_reagent(true_context_smiles[4],null_model_combos[j][4],list_of_metal_atoms,list_of_full_metal_names)

		if all(pred_true_match) and j<=2:
			top3_flag = 1
		if all(pred_true_match):
			top10_flag = 1
			true_context_rank = j
		if all(pred_true_similar) and j<=2:
			top3_similar_flag = 1
		if all(pred_true_similar):
			top10_similar_flag = 1
		if pred_true_match[0] and pred_true_match[1] and pred_true_match[3] and j<=2:
			c1s1r1_top3_flag = 1
		if pred_true_match[0] and pred_true_match[1] and pred_true_match[3]:
			c1s1r1_top10_flag = 1
		if pred_true_similar[0] and pred_true_similar[1] and pred_true_similar[3] and j<=2:
			c1s1r1_similar_top3_flag = 1
		if pred_true_similar[0] and pred_true_similar[1] and pred_true_similar[3]:
			c1s1r1_similar_top10_flag = 1
		if pred_true_match[1] and j<=2:
			s1_top3_flag=1
		if pred_true_match[1]:
			s1_top10_flag = 1
		if pred_true_similar[1] and j<=2:
			s1_top3_similar_flag=1
		if pred_true_similar[1]:
			s1_top10_similar_flag = 1
		if pred_true_match[3] and j<=2:
			r1_top3_flag=1
		if pred_true_match[3]:
			r1_top10_flag = 1
		if pred_true_similar[3] and j<=2:
			r1_top3_similar_flag=1
		if pred_true_similar[3]:
			r1_top10_similar_flag = 1
		if pred_true_match[2] and j<=2:
			s2_top3_flag=1
		if pred_true_match[2]:
			s2_top10_flag = 1
		if pred_true_similar[2] and j<=2:
			s2_top3_similar_flag=1
		if pred_true_similar[2]:
			s2_top10_similar_flag = 1
		if pred_true_match[4] and j<=2:
			r2_top3_flag=1
		if pred_true_match[4]:
			r2_top10_flag = 1
		if pred_true_similar[4] and j<=2:
			r2_top3_similar_flag=1
		if pred_true_similar[4]:
			r2_top10_similar_flag = 1
		if pred_true_match[0] and j<=2:
			c1_top3_flag=1
		if pred_true_match[0]:
			c1_top10_flag = 1
		if pred_true_similar[0] and j<=2:
			c1_top3_similar_flag=1
		if pred_true_similar[0]:
			c1_top10_similar_flag = 1
		
		pred_context_smiles_list.append([pred_context_smiles,pred_true_match,pred_true_similar,sum(pred_true_match),sum(pred_true_similar),context_combos[j][5]])
	
		#####repeat for null model
		if all(null_true_match) and j<=2:
			top3_flag_null = 1
		if all(null_true_match):
			top10_flag_null = 1
			true_context_rank_null = j
		if all(null_true_similar) and j<=2:
			top3_similar_flag_null = 1
		if all(null_true_similar):
			top10_similar_flag_null = 1
		if null_true_match[0] and null_true_match[1] and null_true_match[3] and j<=2:
			c1s1r1_top3_flag_null = 1
		if null_true_match[0] and null_true_match[1] and null_true_match[3]:
			c1s1r1_top10_flag_null = 1
		if null_true_similar[0] and null_true_similar[1] and null_true_similar[3] and j<=2:
			c1s1r1_similar_top3_flag_null = 1
		if null_true_similar[0] and null_true_similar[1] and null_true_similar[3]:
			c1s1r1_similar_top10_flag_null = 1
		if null_true_match[1] and j<=2:
			s1_top3_flag_null=1
		if null_true_match[1]:
			s1_top10_flag_null = 1
		if null_true_similar[1] and j<=2:
			s1_top3_similar_flag_null=1
		if null_true_similar[1]:
			s1_top10_similar_flag_null = 1
		if null_true_match[3] and j<=2:
			r1_top3_flag_null=1
		if null_true_match[3]:
			r1_top10_flag_null = 1
		if null_true_similar[3] and j<=2:
			r1_top3_similar_flag_null=1
		if null_true_similar[3]:
			r1_top10_similar_flag_null = 1
		if null_true_match[2] and j<=2:
			s2_top3_flag_null=1
		if null_true_match[2]:
			s2_top10_flag_null = 1
		if null_true_similar[2] and j<=2:
			s2_top3_similar_flag_null=1
		if null_true_similar[2]:
			s2_top10_similar_flag_null = 1
		if null_true_match[4] and j<=2:
			r2_top3_flag_null=1
		if null_true_match[4]:
			r2_top10_flag_null = 1
		if null_true_similar[4] and j<=2:
			r2_top3_similar_flag_null=1
		if null_true_similar[4]:
			r2_top10_similar_flag_null = 1
		if null_true_match[0] and j<=2:
			c1_top3_flag_null=1
		if null_true_match[0]:
			c1_top10_flag_null = 1
		if null_true_similar[0] and j<=2:
			c1_top3_similar_flag_null=1
		if null_true_similar[0]:
			c1_top10_similar_flag_null = 1
		#####
	top3_ctr += top3_flag
	top10_ctr += top10_flag
	top3_similar_ctr+= top3_similar_flag
	top10_similar_ctr+= top10_similar_flag
	c1s1r1_top3 += c1s1r1_top3_flag
	c1s1r1_top10 += c1s1r1_top10_flag
	c1s1r1_similar_top3 += c1s1r1_similar_top3_flag
	c1s1r1_similar_top10 += c1s1r1_similar_top10_flag
	s1_top3_ctr += s1_top3_flag
	s1_top10_ctr += s1_top10_flag	
	s1_top3_similar_ctr += s1_top3_similar_flag	
	s1_top10_similar_ctr += s1_top10_similar_flag
	r1_top3_ctr += r1_top3_flag
	r1_top10_ctr += r1_top10_flag
	r1_top3_similar_ctr += r1_top3_similar_flag
	r1_top10_similar_ctr += r1_top10_similar_flag
	s2_top3_ctr += s2_top3_flag
	s2_top10_ctr += s2_top10_flag	
	s2_top3_similar_ctr += s2_top3_similar_flag	
	s2_top10_similar_ctr += s2_top10_similar_flag
	r2_top3_ctr += r2_top3_flag
	r2_top10_ctr += r2_top10_flag
	r2_top3_similar_ctr += r2_top3_similar_flag
	r2_top10_similar_ctr += r2_top10_similar_flag
	c1_top3_ctr += c1_top3_flag
	c1_top10_ctr += c1_top10_flag
	c1_top3_similar_ctr += c1_top3_similar_flag
	c1_top10_similar_ctr += c1_top10_similar_flag

	######null model######
	top3_ctr_null += top3_flag_null
	top10_ctr_null += top10_flag_null
	top3_similar_ctr_null+= top3_similar_flag_null
	top10_similar_ctr_null+= top10_similar_flag_null
	c1s1r1_top3 += c1s1r1_top3_flag_null
	c1s1r1_top10 += c1s1r1_top10_flag_null
	c1s1r1_similar_top3 += c1s1r1_similar_top3_flag_null
	c1s1r1_similar_top10 += c1s1r1_similar_top10_flag_null
	s1_top3_ctr_null += s1_top3_flag_null
	s1_top10_ctr_null += s1_top10_flag_null	
	s1_top3_similar_ctr_null += s1_top3_similar_flag_null	
	s1_top10_similar_ctr_null += s1_top10_similar_flag_null
	r1_top3_ctr_null += r1_top3_flag_null
	r1_top10_ctr_null += r1_top10_flag_null
	r1_top3_similar_ctr_null += r1_top3_similar_flag_null
	r1_top10_similar_ctr_null += r1_top10_similar_flag_null
	s2_top3_ctr_null += s2_top3_flag_null
	s2_top10_ctr_null += s2_top10_flag_null	
	s2_top3_similar_ctr_null += s2_top3_similar_flag_null	
	s2_top10_similar_ctr_null += s2_top10_similar_flag_null
	r2_top3_ctr_null += r2_top3_flag_null
	r2_top10_ctr_null += r2_top10_flag_null
	r2_top3_similar_ctr_null += r2_top3_similar_flag_null
	r2_top10_similar_ctr_null += r2_top10_similar_flag_null
	c1_top3_ctr_null += c1_top3_flag_null
	c1_top10_ctr_null += c1_top10_flag_null
	c1_top3_similar_ctr_null += c1_top3_similar_flag_null
	c1_top10_similar_ctr_null += c1_top10_similar_flag_null
	#########

	temp_pred = context_combos[0][5]
	top_pred_context = pred_context_smiles_list[0][0]
	closest_pred = max(pred_context_smiles_list, key = lambda x: x[4])
	closest_pred_context = closest_pred[0]
	closest_pred_temp = closest_pred[-1] 

	rid_list.append(rxn_ids[0])
	rxn_smiles_list.append(rxn_smiles)
	true_context_list.append(true_context_smiles)
	top_pred_list.append(top_pred_context)
	closest_pred_list.append(closest_pred_context)
	true_context_rank_list.append(true_context_rank)
	temp_true_list.append(temp_true)
	temp_pred_list.append(temp_pred)
	closest_temp_pred.append(closest_pred_temp)
	top3_similar_list.append(top3_similar_flag)
	top10_similar_list.append(top10_similar_flag)
	c1s1r1_top3_list.append(c1s1r1_top3_flag)
	c1s1r1_top10_list.append(c1s1r1_top10_flag)
	c1s1r1_similar_top3_list.append(c1s1r1_similar_top3_flag)
	c1s1r1_similar_top10_list.append(c1s1r1_similar_top10_flag)
	s1_top3_list.append(s1_top3_flag)
	s1_top10_list.append(s1_top10_flag)
	s1_top3_similar_list.append(s1_top3_similar_flag)
	s1_top10_similar_list.append(s1_top10_similar_flag)
	s2_top3_list.append(s2_top3_flag)
	s2_top10_list.append(s2_top10_flag)
	s2_top3_similar_list.append(s2_top3_similar_flag)
	s2_top10_similar_list.append(s2_top10_similar_flag)
	r1_top3_list.append(r1_top3_flag)
	r1_top10_list.append(r1_top10_flag)
	r1_top3_similar_list.append(r1_top3_similar_flag)
	r1_top10_similar_list.append(r1_top10_similar_flag)
	r2_top3_list.append(r2_top3_flag)
	r2_top10_list.append(r2_top10_flag)
	r2_top3_similar_list.append(r2_top3_similar_flag)
	r2_top10_similar_list.append(r2_top10_similar_flag)
	c1_top3_list.append(c1_top3_flag)
	c1_top10_list.append(c1_top10_flag)
	c1_top3_similar_list.append(c1_top3_similar_flag)
	c1_top10_similar_list.append(c1_top10_similar_flag)

	true_context_rank_list_null.append(true_context_rank_null)
	top3_similar_list_null.append(top3_similar_flag_null)
	top10_similar_list_null.append(top10_similar_flag_null)
	c1s1r1_top3_list_null.append(c1s1r1_top3_flag_null)
	c1s1r1_top10_list_null.append(c1s1r1_top10_flag_null)
	c1s1r1_similar_top3_list_null.append(c1s1r1_similar_top3_flag_null)
	c1s1r1_similar_top10_list_null.append(c1s1r1_similar_top10_flag_null)
	s1_top3_list_null.append(s1_top3_flag_null)
	s1_top10_list_null.append(s1_top10_flag_null)
	s1_top3_similar_list_null.append(s1_top3_similar_flag_null)
	s1_top10_similar_list_null.append(s1_top10_similar_flag_null)
	s2_top3_list_null.append(s2_top3_flag_null)
	s2_top10_list_null.append(s2_top10_flag_null)
	s2_top3_similar_list_null.append(s2_top3_similar_flag_null)
	s2_top10_similar_list_null.append(s2_top10_similar_flag_null)
	r1_top3_list_null.append(r1_top3_flag_null)
	r1_top10_list_null.append(r1_top10_flag_null)
	r1_top3_similar_list_null.append(r1_top3_similar_flag_null)
	r1_top10_similar_list_null.append(r1_top10_similar_flag_null)
	r2_top3_list_null.append(r2_top3_flag_null)
	r2_top10_list_null.append(r2_top10_flag_null)
	r2_top3_similar_list_null.append(r2_top3_similar_flag_null)
	r2_top10_similar_list_null.append(r2_top10_similar_flag_null)
	c1_top3_list_null.append(c1_top3_flag_null)
	c1_top10_list_null.append(c1_top10_flag_null)
	c1_top3_similar_list_null.append(c1_top3_similar_flag_null)
	c1_top10_similar_list_null.append(c1_top10_similar_flag_null)
	# true_context_smiles = '.'.join(true_context_smiles)
	# top_pred_context = '.'.join(top_pred_context)
	# closest_pred_context = '.'.join(closest_pred_context)
	# results_df.loc[i] = [rxn_ids[0],
	# 					# rxn_smiles, true_context_smiles, top_pred_context, closest_pred_context, 
	# 					true_context_rank, temp_true, temp_pred, closest_pred_temp,
	# 						top3_similar_flag, top10_similar_flag, c1s1r1_top3_flag, c1s1r1_top10_flag, c1s1r1_similar_top3_flag, c1s1r1_similar_top10_flag,
	# 						s1_top3_flag, s1_top10_flag, r1_top3_flag, r1_top10_flag, s1_top3_similar_flag, s1_top10_similar_flag, r1_top3_similar_flag,
	# 						r1_top10_similar_flag, s2_top3_flag, s2_top10_flag, r2_top3_flag, r2_top10_flag, s2_top3_similar_flag, s2_top10_similar_flag, 
	# 						r2_top3_similar_flag, r2_top10_similar_flag, c1_flag, c1_similar_flag
	# 						]
	# ctr+=1
	# outfile.write('rxn_id:{}\t rxn_smiles:{}\t true_context_smiles:{}\t pred_context_smiles:{}\t true_context_rank:{}\t temp_true:{}\t temp_pred:{}\t \n'.format(
	# 	rxn_ids[0],rxn_smiles, true_context_smiles,closest_pred_context,true_context_rank, temp_true, temp_pred))
# top3_acc = float(top3_ctr)/ctr
# top10_acc = float(top10_ctr)/ctr
# top3_similar_acc = float(top3_similar_ctr)/ctr
# top10_similar_acc = float(top10_similar_ctr)/ctr

# print(top3_acc,top10_acc, top3_similar_acc, top10_similar_acc)
# summary_file.write("top3 accuracy: {}\t top10 accuracy: {}\t parital top3 accuracy: {}\t parital top10 accuracy: {}\t\n".format(top3_acc,top10_acc, top3_similar_acc, top10_similar_acc))
# outfile.close()
# results_df = pd.DataFrame(columns = ['rxn_id',
# 									# 'rxn_smiles', 'true_context_smiles', 'top_pred','closest_pred', 
# 									'true_context_rank', 'temp_true', 'temp_pred','closest_pred_temp',
# 									'top3_similar', 'top10_similar', 'c1s1r1_top3', 'c1s1r1_top10', 'c1s1r1_similar_top3', 'c1s1r1_similar_top10',
# 									's1_top3', 's1_top10', 'r1_top3', 'r1_top10', 's1_similar_top3', 's1_similar_top10', 'r1_similar_top3',
# 									'r1_similar_top10','s2_top3', 's2_top10', 'r2_top3', 'r2_top10', 's2_similar_top3', 's2_similar_top10', 'r2_similar_top3',
# 									'r2_similar_top10', 'c1', 'c1_similar'
# 									],
# 						index = range(data['test_nb_samples']))

results_df = pd.DataFrame({
	'rxn_id':rid_list,
	'rxn_smiles':rxn_smiles_list,
	'true_context_smiles':true_context_list,
	'top_pred':top_pred_list,
	'closest_pred':closest_pred_list,
	'true_context_rank':true_context_rank_list,
	'temp_true':temp_true_list,
	'temp_pred':temp_pred_list,
	'closest_pred_temp':closest_temp_pred,
	'top3_similar':top3_similar_list,
	'top10_similar':top10_similar_list,
	'c1s1r1_top3':c1s1r1_top3_list,
	'c1s1r1_top10':c1s1r1_top10_list,
	'c1s1r1_similar_top3':c1s1r1_similar_top3_list,
	'c1s1r1_similar_top10':c1s1r1_similar_top10_list,
	's1_top3':s1_top3_list,
	's1_top10':s1_top10_list,
	's1_similar_top3':s1_top3_similar_list,
	's1_similar_top10':s1_top10_similar_list,
	's2_top3':s2_top3_list,
	's2_top10':s2_top10_list,
	's2_similar_top3':s2_top3_similar_list,
	's2_similar_top10':s2_top10_similar_list,
	'r1_top3':r1_top3_list,
	'r1_top10':r1_top10_list,
	'r1_similar_top3':r1_top3_similar_list,
	'r1_similar_top10':r1_top10_similar_list,
	'r2_top3':r2_top3_list,
	'r2_top10':r2_top10_list,
	'r2_similar_top3':r2_top3_similar_list,
	'r2_similar_top10':r2_top10_similar_list,
	'c1_top3':c1_top3_list,
	'c1_top10':c1_top10_list,
	'c1_similar_top3':c1_top3_similar_list,
	'c1_similar_top10':c1_top10_similar_list,

	'true_context_rank_null':true_context_rank_list_null,
	'top3_similar_null':top3_similar_list_null,
	'top10_similar_null':top10_similar_list_null,
	'c1s1r1_top3_null':c1s1r1_top3_list_null,
	'c1s1r1_top10_null':c1s1r1_top10_list_null,
	'c1s1r1_similar_top3_null':c1s1r1_similar_top3_list_null,
	'c1s1r1_similar_top10_null':c1s1r1_similar_top10_list_null,
	's1_top3_null':s1_top3_list_null,
	's1_top10_null':s1_top10_list_null,
	's1_similar_top3_null':s1_top3_similar_list_null,
	's1_similar_top10_null':s1_top10_similar_list_null,
	's2_top3_null':s2_top3_list_null,
	's2_top10_null':s2_top10_list_null,
	's2_similar_top3_null':s2_top3_similar_list_null,
	's2_similar_top10_null':s2_top10_similar_list_null,
	'r1_top3_null':r1_top3_list_null,
	'r1_top10_null':r1_top10_list_null,
	'r1_similar_top3_null':r1_top3_similar_list_null,
	'r1_similar_top10_null':r1_top10_similar_list_null,
	'r2_top3_null':r2_top3_list_null,
	'r2_top10_null':r2_top10_list_null,
	'r2_similar_top3_null':r2_top3_similar_list_null,
	'r2_similar_top10_null':r2_top10_similar_list_null,
	'c1_top3_null':c1_top3_list_null,
	'c1_top10_null':c1_top10_list_null,
	'c1_similar_top3_null':c1_top3_similar_list_null,
	'c1_similar_top10_null':c1_top10_similar_list_null,
	})

# results_df.to_csv('all_results_partial_fixed_ranking.csv')


df_agg_repeat_rxn = results_df.groupby('rxn_id').agg({'true_context_rank':'min','top3_similar':'max','top10_similar':'max',
													'c1s1r1_top3':'max','c1s1r1_top10':'max', 'c1s1r1_similar_top3':'max', 'c1s1r1_similar_top10':'max',
													's1_top3':'max','s1_top10':'max', 's1_similar_top3':'max', 's1_similar_top10':'max',
													'r1_top3':'max','r1_top10':'max', 'r1_similar_top3':'max', 'r1_similar_top10':'max',
													's2_top3':'max','s2_top10':'max', 's2_similar_top3':'max', 's2_similar_top10':'max',
													'r2_top3':'max','r2_top10':'max', 'r2_similar_top3':'max', 'r2_similar_top10':'max',
													'c1_top3':'max','c1_top10':'max', 'c1_similar_top3':'max', 'c1_similar_top10':'max',
													'true_context_rank_null':'min','top3_similar_null':'max','top10_similar_null':'max',
													'c1s1r1_top3_null':'max','c1s1r1_top10_null':'max', 'c1s1r1_similar_top3_null':'max', 'c1s1r1_similar_top10_null':'max',
													's1_top3_null':'max','s1_top10_null':'max', 's1_similar_top3_null':'max', 's1_similar_top10_null':'max',
													'r1_top3_null':'max','r1_top10_null':'max', 'r1_similar_top3_null':'max', 'r1_similar_top10_null':'max',
													's2_top3_null':'max','s2_top10_null':'max', 's2_similar_top3_null':'max', 's2_similar_top10_null':'max',
													'r2_top3_null':'max','r2_top10_null':'max', 'r2_similar_top3_null':'max', 'r2_similar_top10_null':'max',
													'c1_top3_null':'max','c1_top10_null':'max', 'c1_similar_top3_null':'max', 'c1_similar_top10_null':'max',
													})
top3_acc = df_agg_repeat_rxn[df_agg_repeat_rxn['true_context_rank']<=2].count()/df_agg_repeat_rxn.count()
top10_acc = df_agg_repeat_rxn[df_agg_repeat_rxn['true_context_rank']<=9].count()/df_agg_repeat_rxn.count()
parital_top3_acc = df_agg_repeat_rxn['top3_similar'].mean()
parital_top10_acc = df_agg_repeat_rxn['top10_similar'].mean()
c1s1r1_top3_acc = df_agg_repeat_rxn['c1s1r1_top3'].mean()
c1s1r1_top10_acc = df_agg_repeat_rxn['c1s1r1_top10'].mean()
c1s1r1_similar_top3_acc = df_agg_repeat_rxn['c1s1r1_similar_top3'].mean()
c1s1r1_similar_top10_acc = df_agg_repeat_rxn['c1s1r1_similar_top10'].mean()
s1_top3_acc = df_agg_repeat_rxn['s1_top3'].mean()
s1_top10_acc = df_agg_repeat_rxn['s1_top10'].mean()
s1_similar_top3_acc = df_agg_repeat_rxn['s1_similar_top3'].mean()
s1_similar_top10_acc = df_agg_repeat_rxn['s1_similar_top10'].mean()
r1_top3_acc = df_agg_repeat_rxn['r1_top3'].mean()
r1_top10_acc = df_agg_repeat_rxn['r1_top10'].mean()
r1_similar_top3_acc = df_agg_repeat_rxn['r1_similar_top3'].mean()
r1_similar_top10_acc = df_agg_repeat_rxn['r1_similar_top10'].mean()
s2_top3_acc = df_agg_repeat_rxn['s2_top3'].mean()
s2_top10_acc = df_agg_repeat_rxn['s2_top10'].mean()
s2_similar_top3_acc = df_agg_repeat_rxn['s2_similar_top3'].mean()
s2_similar_top10_acc = df_agg_repeat_rxn['s2_similar_top10'].mean()
r2_top3_acc = df_agg_repeat_rxn['r2_top3'].mean()
r2_top10_acc = df_agg_repeat_rxn['r2_top10'].mean()
r2_similar_top3_acc = df_agg_repeat_rxn['r2_similar_top3'].mean()
r2_similar_top10_acc = df_agg_repeat_rxn['r2_similar_top10'].mean()
c1_top3_acc = df_agg_repeat_rxn['c1_top3'].mean()
c1_top10_acc = df_agg_repeat_rxn['c1_top10'].mean()
c1_similar_top3_acc = df_agg_repeat_rxn['c1_similar_top3'].mean()
c1_similar_top10_acc = df_agg_repeat_rxn['c1_similar_top10'].mean()

top3_acc_null = df_agg_repeat_rxn[df_agg_repeat_rxn['true_context_rank_null']<=2].count()/df_agg_repeat_rxn.count()
top10_acc_null = df_agg_repeat_rxn[df_agg_repeat_rxn['true_context_rank_null']<=9].count()/df_agg_repeat_rxn.count()
parital_top3_acc_null = df_agg_repeat_rxn['top3_similar_null'].mean()
parital_top10_acc_null = df_agg_repeat_rxn['top10_similar_null'].mean()
c1s1r1_top3_acc_null = df_agg_repeat_rxn['c1s1r1_top3_null'].mean()
c1s1r1_top10_acc_null = df_agg_repeat_rxn['c1s1r1_top10_null'].mean()
c1s1r1_similar_top3_acc_null = df_agg_repeat_rxn['c1s1r1_similar_top3_null'].mean()
c1s1r1_similar_top10_acc_null = df_agg_repeat_rxn['c1s1r1_similar_top10_null'].mean()
s1_top3_acc_null = df_agg_repeat_rxn['s1_top3_null'].mean()
s1_top10_acc_null = df_agg_repeat_rxn['s1_top10_null'].mean()
s1_similar_top3_acc_null = df_agg_repeat_rxn['s1_similar_top3_null'].mean()
s1_similar_top10_acc_null = df_agg_repeat_rxn['s1_similar_top10_null'].mean()
r1_top3_acc_null = df_agg_repeat_rxn['r1_top3_null'].mean()
r1_top10_acc_null = df_agg_repeat_rxn['r1_top10_null'].mean()
r1_similar_top3_acc_null = df_agg_repeat_rxn['r1_similar_top3_null'].mean()
r1_similar_top10_acc_null = df_agg_repeat_rxn['r1_similar_top10_null'].mean()
s2_top3_acc_null = df_agg_repeat_rxn['s2_top3_null'].mean()
s2_top10_acc_null = df_agg_repeat_rxn['s2_top10_null'].mean()
s2_similar_top3_acc_null = df_agg_repeat_rxn['s2_similar_top3_null'].mean()
s2_similar_top10_acc_null = df_agg_repeat_rxn['s2_similar_top10_null'].mean()
r2_top3_acc_null = df_agg_repeat_rxn['r2_top3_null'].mean()
r2_top10_acc_null = df_agg_repeat_rxn['r2_top10_null'].mean()
r2_similar_top3_acc_null = df_agg_repeat_rxn['r2_similar_top3_null'].mean()
r2_similar_top10_acc_null = df_agg_repeat_rxn['r2_similar_top10_null'].mean()
c1_top3_acc_null = df_agg_repeat_rxn['c1_top3_null'].mean()
c1_top10_acc_null = df_agg_repeat_rxn['c1_top10_null'].mean()
c1_similar_top3_acc_null = df_agg_repeat_rxn['c1_similar_top3_null'].mean()
c1_similar_top10_acc_null = df_agg_repeat_rxn['c1_similar_top10_null'].mean()

df_agg_repeat_rxn.to_csv('all_results_partial_fixed_ranking_subset.csv')

print('top3:{}\n top10:{}\n top3_similar:{}\n top10_similar:{}\n c1s1r1_top3:{}\n c1s1r1_top10:{}\n c1s1r1_similar_top3:{}\n \
	c1s1r1_similar_top10:{}\n, s1_top3:{}\n, s1_top10:{}\n, s1_similar_top3:{}\n, s1_similar_top10:{}\n, r1_top3:{}\n, r1_top10:{}\n \
	r1_similar_top3:{}\n, r1_similar_top10:{}\n s2_top3:{}\n, s2_top10:{}\n, s2_similar_top3:{}\n, s2_similar_top10:{}\n, r2_top3:{}\n, r2_top10:{}\n \
	r2_similar_top3:{}\n, r2_similar_top10:{}\n, c1_top3:{}\n, c1_top10:{}\n c1_similar_top3:{}\n, c1_similar_top10:{}\n'.format(top3_acc[0], top10_acc[0], parital_top3_acc,parital_top10_acc, c1s1r1_top3_acc,\
		c1s1r1_top10_acc,c1s1r1_similar_top3_acc,c1s1r1_similar_top10_acc,s1_top3_acc,s1_top10_acc,s1_similar_top3_acc,s1_similar_top10_acc,\
		r1_top3_acc,r1_top10_acc,r1_similar_top3_acc, r1_similar_top10_acc,s2_top3_acc,s2_top10_acc,s2_similar_top3_acc,s2_similar_top10_acc,\
		r2_top3_acc,r2_top10_acc,r2_similar_top3_acc, r2_similar_top10_acc, c1_top3_acc,c1_top10_acc,c1_similar_top3_acc, c1_similar_top10_acc
	)
	)

print('For NULL MODEL:: top3:{}\n top10:{}\n top3_similar:{}\n top10_similar:{}\n c1s1r1_top3:{}\n c1s1r1_top10:{}\n c1s1r1_similar_top3:{}\n \
	c1s1r1_similar_top10:{}\n, s1_top3:{}\n, s1_top10:{}\n, s1_similar_top3:{}\n, s1_similar_top10:{}\n, r1_top3:{}\n, r1_top10:{}\n \
	r1_similar_top3:{}\n, r1_similar_top10:{}\n s2_top3:{}\n, s2_top10:{}\n, s2_similar_top3:{}\n, s2_similar_top10:{}\n, r2_top3:{}\n, r2_top10:{}\n \
	r2_similar_top3:{}\n, r2_similar_top10:{}\n, c1_top3:{}\n, c1_top10:{}\n c1_similar_top3:{}\n, c1_similar_top10:{}\n'.format(top3_acc_null[0], top10_acc_null[0], parital_top3_acc_null,parital_top10_acc_null, c1s1r1_top3_acc_null,\
		c1s1r1_top10_acc_null,c1s1r1_similar_top3_acc_null,c1s1r1_similar_top10_acc_null,s1_top3_acc_null,s1_top10_acc_null,s1_similar_top3_acc_null,s1_similar_top10_acc_null,\
		r1_top3_acc_null,r1_top10_acc_null,r1_similar_top3_acc_null, r1_similar_top10_acc_null,s2_top3_acc_null,s2_top10_acc_null,s2_similar_top3_acc_null,s2_similar_top10_acc_null,\
		r2_top3_acc_null,r2_top10_acc_null,r2_similar_top3_acc_null, r2_similar_top10_acc_null, c1_top3_acc_null,c1_top10_acc_null,c1_similar_top3_acc_null, c1_similar_top10_acc_null
	)
	)

# df_correct_pred = results_df.loc[(results_df['true_context_rank']==0) & (results_df['temp_true']!=-1.0)]
df_T = results_df.loc[(results_df['temp_true']!=-1.0)]
df_T_correct = df_T.loc[(df_T['true_context_rank']<=9)]

df_T['T_diff'] = np.abs(df_T['temp_pred']-df_T['temp_true'])
df_T['T_diff_10_ind'] = df_T['T_diff']<=10
df_T['T_diff_20_ind'] = df_T['T_diff']<=20
df_T_agg = df_T.groupby('rxn_id').agg({'T_diff_10_ind':'max','T_diff_20_ind':'max'})


temp_10K_error = df_T[df_T['T_diff_10_ind']].count()/df_T.count()
temp_20K_error = df_T[df_T['T_diff_20_ind']].count()/df_T.count()
print(temp_10K_error[0],temp_20K_error[0])
df_T_correct['T_diff'] = np.abs(df_T_correct['closest_pred_temp']-df_T_correct['temp_true'])
df_T_correct['T_diff_10_ind'] = df_T_correct['T_diff']<=10
df_T_correct['T_diff_20_ind'] = df_T_correct['T_diff']<=20

df_T_correct_agg = df_T_correct.groupby('rxn_id').agg({'T_diff_10_ind':'max','T_diff_20_ind':'max'})

temp_10K_error = df_T_correct[df_T_correct['T_diff_10_ind']].count()/df_T_correct.count()
temp_20K_error = df_T_correct[df_T_correct['T_diff_20_ind']].count()/df_T_correct.count()
print(temp_10K_error[0],temp_20K_error[0])



print(temp_10K_error[0],temp_20K_error[0])
temp_mse = np.mean(df_correct_pred['T_diff'])