import pandas as pd
import pickle
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from pymongo import MongoClient
from scipy.spatial.distance import pdist
from adjustText import adjust_text

def plot_embedding(X, label, color_list=None, title=None):
	x_min, x_max = np.min(X, 0), np.max(X, 0)
	X = (X - x_min) / (x_max - x_min)
	# cmap = sns.cubehelix_palette(as_cmap=True)


	label_set = list(set(color_list))
	color_set = ['C'+str(i) for i in range(len(label_set))]
	color_dict = dict(zip(label_set, color_set))
	colors=[color_dict[key] for key in color_list]
# 	print(len(colors))
	df = pd.DataFrame({'x':X[:,0].flatten(),
					   'y':X[:,1].flatten(),
					   'color':colors,
					   'label':color_list,
					   })
	# rint(label_set, color_set, color_dict, colors)
	plt.figure()
	f, ax = plt.subplots(figsize=(10,10))
	for i, df_by_color in df.groupby('color'):
# 		print(i)
		points = ax.scatter(df_by_color['x'],df_by_color['y'], c=df_by_color['color'], label = label_set[color_set.index(i)])
	handles, labels = ax.get_legend_handles_labels()
# 	print(labels)
	force_order = ['Non-polar','Polar non-protic','Polar protic','Halogenated']
	order_vec = [force_order.index(item) for item in labels]
	handles, labels, order_vec = zip(*sorted(zip(handles, labels, order_vec), key = lambda x: x[2]))
	ax.legend(handles, labels)
	# 
	# f.colorbar(points)
	texts=[]
	for i, txt in enumerate(label):
		# ax.annotate(txt, (X[i,0],X[i,1]), fontsize = 7)
		texts.append(plt.text(X[i,0],X[i,1],txt,fontsize=10))
	# # for i in range(X.shape[0]):
	#     plt.text(X[i, 0], X[i, 1], '',
	             # color=plt.cm.Set1(y[i] / 10.),
	             # fontdict={'weight': 'bold', 'size': 9})
	# if hasattr(offsetbox, 'AnnotationBbox'):
	#     # only print thumbnails with matplotlib > 1.0
	#     shown_images = np.array([[1., 1.]])  # just something big
	#     for i in range(digits.data.shape[0]):
	#         dist = np.sum((X[i] - shown_images) ** 2, 1)
	#         if np.min(dist) < 4e-3:
	#             # don't show points that are too close
	#             continue
	#         shown_images = np.r_[shown_images, [X[i]]]
	#         imagebox = offsetbox.AnnotationBbox(
	#             offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
	#             X[i])
	#         ax.add_artist(imagebox)
	plt.xticks([]), plt.yticks([])
	if title is not None:
	    plt.title(title)

	adjust_text(texts, only_move={'text':'y'}, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))
	plt.show()
solv_sim_cal = solvent_similarity_calculator()

####IMPORTANT####
####change model_path, dict_path, weights_path to the location of those files###
solv_sim_cal.load(model_path='', dict_path='', weights_path='')

# s1_dict_file = gc.NEURALNET_CONTEXT_REC['info_path'] + "s1_dict.pickle"
# with open(s1_dict_file, "r") as S1_DICT_F:
# 	s1_dict = pickle.load(S1_DICT_F)

# print s1_dict

slv_list = [
			'ClCCl',
			'C1CCOC1',
			'O',
			'CN(C)C=O',
			'CCO',
			'CO',
			'Cc1ccccc1',
			'CC#N',
			'CCOCC',
			'C1COCCO1',
			'c1ccccc1',
			'CC(C)=O',
			'ClC(Cl)Cl',
			'CS(C)=O',
			'CCOC(C)=O',
			'CC(=O)O',
			'ClCCCl',
			'CCCCCC',
			'c1ccncc1',
			'CC(C)O',
			'COCCOC',
			'ClC(Cl)(Cl)Cl',
			'CC(C)(C)O',
			'CN1CCCC1=O',
			'CC(=O)N(C)C',
			'Cc1ccccc1C',
			'CCCCO',
			'ClC1=CC=CC=C1',
			'Reaxys Name mineral oil',
			'CCCCC',
			'C1CCCCC1',
			'Reaxys Name chloroform-d1',
			'CNC(C)=O',
			'C[N+](=O)[O-]',
			'CN(C)P(=O)(N(C)C)N(C)C',
			'OCCO',
			'CCC(C)=O',
			'ClC1=C(Cl)C=CC=C1',
			'[2H]C1=C([2H])C([2H])=C([2H])C([2H])=C1[2H]',
			'CC(=O)OC(C)=O',
			'Reaxys Name Petroleum ether',
			'Reaxys Name aq$dot$ phosphate buffer',
			'Cl',
			'O=C(O)C(F)(F)F',
			'CCCCCCC',
			'COCCOCCOC',
			'c1ccc(Oc2ccccc2)cc1',
			'N',
			'O=S(=O)(O)O',
			'CCCO',
			]

num_slv = len(slv_list)
slv_fp_mtx = np.zeros([len(slv_list),300])
column_names = [
'Dichloromethane',
'THF',
'Water',
'DMF',
'Ethanol',
'Methanol',
'Toluene',
'Acetonitrile',
'Diethyl ether',
'1,4-dioxane',
'Benzene',
'Acetone',
'Chloroform',
'DMSO',
'Ethyl acetate',
'Acetic acid',
'1,2-dichloroethane',
'Hexane',
'Pyridine',
'Isopropanol',
'1,2-dimethoxyethane (DME)',
'Carbon tetrachloride',
'Tert-butanol',
'N-methylpyrrolidone',
'Dimethylacetamide',
'Xylene',
'N-butanol',
'Chlorobenzene',
'Mineral oil',
'Pentane',
'Cyclohexane',
'Chloroform-d1',
'N-methylacetamide',
'Nitromethane',
'Hexamethylphosphoramide (HMPA)',
'Ethylene glycol',
'Methyl ethyl ketone',
'1,2-dichlorobenzene',
'Benzene-d6',
'Acetic anhydride',
'Petroleum ether',
'Aqueous phosphate buffer',
'Hydrogen chloride',
'Trifluoroacetic acid (TFA)',
'Heptane',
'Diglyme',
'Diphenyl ether',
'Ammonia',
'Sufuric acid',
'1-propanol'
]

for sid,slv in enumerate(slv_list):
	if True:
		slv_fp = solv_sim_cal.get_slv_fp(slv)
		norm_slv_fp = slv_fp/np.linalg.norm(slv_fp)
		slv_fp_mtx[sid,:] = norm_slv_fp
		# column_names.append(slv)



slv_fp_embedded = TSNE(n_components = 2,  metric = 'euclidean', n_iter = 1000000, learning_rate = 10, random_state=9999).fit_transform(slv_fp_mtx)


slv_fp_embedded.shape


# slv_df = pd.read_csv('slv_data_with_dc.csv')
# dc = slv_df['dielec_const']
# print(slv_fp_mtx.shape)
slv_labels = [
'Halogenated',
'Polar non-protic',
'Polar protic',
'Polar non-protic',
'Polar protic',
'Polar protic',
'Non-polar',
'Polar non-protic',
'Polar non-protic',
'Polar non-protic',
'Non-polar',
'Polar non-protic',
'Halogenated',
'Polar non-protic',
'Polar non-protic',
'Polar protic',
'Halogenated',
'Non-polar',
'Polar non-protic',
'Polar protic',
'Polar non-protic',
'Halogenated',
'Polar protic',
'Polar non-protic',
'Polar non-protic',
'Non-polar',
'Polar protic',
'Halogenated',
'Non-polar',
'Non-polar',
'Non-polar',
'Halogenated',
'Polar protic',
'Polar non-protic',
'Polar non-protic',
'Polar protic',
'Polar non-protic',
'Halogenated',
'Non-polar',
'Polar non-protic',
'Non-polar',
'Polar protic',
'Polar protic',
'Polar protic',
'Non-polar',
'Polar non-protic',
'Polar non-protic',
'Polar protic',
'Polar protic',
'Polar protic'
]

import pandas as pd
slv_df = pd.DataFrame({'name':column_names,
					   'slv_embedded_1':slv_fp_embedded[:,0],
					   'slv_embedded_2':slv_fp_embedded[:,1],
					   # 'slv_c':slv_c_param,
# 					   'slv_e':slv_e_param,
# 					   'slv_s':slv_s_param,
# 					   'slv_a':slv_a_param,
# 					   'slv_b':slv_b_param,
# 					   'slv_v':slv_v_param,
					   'smiles':slv_list,
					   'slv_labels':slv_labels,
					   })
slv_df.to_csv('slv_embedding.csv')
plot_embedding(slv_fp_embedded, label = column_names, color_list = slv_labels)