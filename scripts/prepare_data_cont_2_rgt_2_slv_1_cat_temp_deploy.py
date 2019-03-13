from pymongo import MongoClient
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import pickle
import pandas as pd
from tqdm import tqdm
from multiprocessing import Manager, Queue, Process
import Queue as VanillaQueue
import time
from scipy import sparse
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import itertools, operator
from collections import Counter
import datetime
from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)
###open database 


client = #######

db = client['reaxys_v2']

instance_db = db['instances']
reaction_db = db['reactions']
chemical_db = db['chemicals']
def create_rxn_Morgan2FP(rsmi, psmi, rxnfpsize = 16384, pfpsize = 16384, useFeatures=False,calculate_rfp = True):
    """Create a rxn Morgan (r=2) fingerprint as bit vector from SMILES string lists of reactants and products"""
    # Modified from Schneider's code (2014)
    if calculate_rfp is True:
	    rsmi = rsmi.encode('utf-8')
	    try:
	    	mol = Chem.MolFromSmiles(rsmi,)
	    except Exception as e:
	    	
	    	return
	    # if mol is None:
	    # 	print(react)
	    try:
	        fp_bit = AllChem.GetMorganFingerprintAsBitVect(mol=mol, radius=2, nBits = rxnfpsize, useFeatures=useFeatures, useChirality = True)
	        fp = np.empty(rxnfpsize,dtype = 'int8')
	        DataStructs.ConvertToNumpyArray(fp_bit,fp)
	        # print(fp.dtype)
	        # fp = np.asarray(fp_bit)
	        # fp = AllChem.GetMorganFingerprint(mol=mol, radius=2, useFeatures=useFeatures)

	    except Exception as e:
	        print("Cannot build reactant fp due to {}".format(e))

	        return
	        
	    rfp = fp
    else:
	    rfp = None

    psmi = psmi.encode('utf-8')
    try:
    	mol = Chem.MolFromSmiles(psmi)
    except Exception as e:
    	# print(psmi)
    	return
    # if mol is None:
    # 	print(product)
    try:
        fp_bit = AllChem.GetMorganFingerprintAsBitVect(mol=mol, radius=2, nBits = pfpsize, useFeatures=useFeatures, useChirality = True)
        fp = np.empty(pfpsize,dtype = 'int8')
        DataStructs.ConvertToNumpyArray(fp_bit,fp)
        # fp = np.asarray(fp_bit)
        # fp = AllChem.GetMorganFingerprint(mol=mol, radius=2, useFeatures=useFeatures)

    except Exception as e:
    	print("Cannot build product fp due to {}".format(e))
    	return
        
    pfp = fp
    # pfp_for_rxn = pfp
    # for product in psmi:
    # 	product = product.encode('utf-8')
    #     mol = Chem.MolFromSmiles(product)
    #     if mol is None:
    #     	print(product)
    #     try:
    #         fp = np.array(
    #             AllChem.GetMorganFingerprintAsBitVect(mol=mol, radius=2, nBits=rxnfpsize, useFeatures=useFeatures))
    #     except Exception as e:
    #         print("Cannot build product fp due to {}".format(e))
    #     if pfp_for_rxn is None:
    #         pfp_for_rxn = fp
    #     else:
    #         pfp_for_rxn += fp
    # if pfp_for_rxn is not None and rfp is not None:
    #     rxnfp = pfp_for_rxn - rfp
    return [pfp, rfp]

##USEFUL UTILITIES
def string_or_range_to_float(text):
	try:
		return float(text)
	except Exception as e:
		x = [z for z in text.strip().split('-') if z not in [u'', u' ']]
		if text.count('-') == 1: # 20 - 30
			try:
				return (float(x[0]) + float(x[1])) / 2.0
			except Exception as e:
				print('Could not convert {}, {}'.format(text, x))
				#print(e)
		elif text.count('-') == 2: # -20 - 0
			try:
				return (-float(x[0]) + float(x[1])) / 2.0
			except Exception as e:
				print('Could not convert {}, {}'.format(text, x))
				#print(e)
		elif text.count('-') == 3: # -20 - -10
			try:
				return (-float(x[0]) - float(x[1])) / 2.0
			except Exception as e:
				print('Could not convert {}, {}'.format(text, x))
				#print(e)
		else:
			print('Could not convert {}'.format(text))
			print(e)
	return np.nan

# def summarize_reaction_outcome(mols, outcome):

# 	h_lost = []
# 	h_gain = []
# 	bond_lost = []
# 	bond_gain = []
# 	try:
# 		conserved_maps = [a.GetProp('molAtomMapNumber') for a in outcome.GetAtoms() if a.HasProp('molAtomMapNumber')]
# 		changes = 0

# 		for atom_prev in mols.GetAtoms():
# 			atomMapNumber = atom_prev.GetProp('molAtomMapNumber')
# 			atom_new = [a for a in outcome.GetAtoms() if a.HasProp('molAtomMapNumber') and a.GetProp('molAtomMapNumber') == atomMapNumber]
# 			if not atom_new: continue
# 			atom_new = atom_new[0]
			
# 			Hs_prev = atom_prev.GetTotalNumHs()
# 			Hs_new  = atom_new.GetTotalNumHs()
# 			if Hs_prev < Hs_new:
# 				#print('    atom {} gained {} hydrogens'.format(atomMapNumber, Hs_new - Hs_prev))
# 				for i in range(Hs_prev, Hs_new):
# 					h_gain.append(atomMapNumber)
# 					changes += 1
# 			if Hs_prev > Hs_new:
# 				#print('    atom {} lost {} hydrogens'.format(atomMapNumber, Hs_prev - Hs_new))
# 				for i in range(Hs_new, Hs_prev): 
# 					h_lost.append(atomMapNumber)
# 					changes += 1

# 			# Charge_prev = atom_prev.GetFormalCharge()
# 			# Charge_new = atom_new.GetFormalCharge()
# 			# if Charge_prev != Charge_new:
# 			# 	#print('    atom {} changed charge ({} to {})'.format(atomMapNumber, Charge_prev, Charge_new))
# 			# 	changes += 1

# 		bonds_prev = {}
# 		for bond in mols.GetBonds():
# 			nums = sorted(
# 				[bond.GetBeginAtom().GetProp('molAtomMapNumber'),
# 				bond.GetEndAtom().GetProp('molAtomMapNumber')])
# 			if (nums[0] not in conserved_maps) and (nums[1] not in conserved_maps): continue
# 			bonds_prev['{}~{}'.format(nums[0], nums[1])] = bond.GetBondTypeAsDouble()
# 		bonds_new = {}
# 		for bond in outcome.GetBonds():
# 			nums = sorted(
# 				[bond.GetBeginAtom().GetProp('molAtomMapNumber'),
# 				bond.GetEndAtom().GetProp('molAtomMapNumber')])
# 			bonds_new['{}~{}'.format(nums[0], nums[1])] = bond.GetBondTypeAsDouble()
		
# 		# print('prev: {}'.format(Chem.MolToSmarts(mols)))
# 		# print('new: {}'.format(Chem.MolToSmarts(outcome)))
# 		# print('bonds_prev: {}'.format(bonds_prev))
# 		# print('bonds_new: {}'.format(bonds_new))

# 		for bond in bonds_prev:
# 			if bond not in bonds_new:
# 				#print('    lost bond {}, order {}'.format(bond, bonds_prev[bond]))
# 				bond_lost.append((bond.split('~')[0], bond.split('~')[1], bonds_prev[bond]))
# 				changes += 1
# 			else:
# 				if bonds_prev[bond] != bonds_new[bond]:
# 					#print('    changed bond {} from order {} to {}'.format(bond, bonds_prev[bond], bonds_new[bond]))
# 					bond_lost.append((bond.split('~')[0], bond.split('~')[1], bonds_prev[bond]))
# 					bond_gain.append((bond.split('~')[0], bond.split('~')[1], bonds_new[bond]))
# 					changes += 1
# 		for bond in bonds_new:
# 			if bond not in bonds_prev:
# 				#print('    new bond {}, order {}'.format(bond, bonds_new[bond]))
# 				bond_gain.append((bond.split('~')[0], bond.split('~')[1], bonds_new[bond]))
# 				changes += 1
# 	except:
# 		pass
# 	#print('{} total changes'.format(changes))
# 	return (sorted(h_lost), sorted(h_gain), sorted(bond_lost), sorted(bond_gain))


#### define workers to process documents
def processing_worker(wid, doc_queue, results_queue, done, exception_count):
	## use exception count to record # of cases where 
	#0. converting smiles to mol fail
	#1. edits generation fail
	#2. edits are None
	
	while True:
		try:
			(rxn_id, rxn_smiles, rgt, slv, cat, temp, yd) = doc_queue.get()
			# print(rxn_id)

			if rxn_id is None: #poison pill
				print("worker {} saw done signal, stopping...".format(wid))
				done[wid] = 1
				break
			pfp_batch = []
			rfp_batch = []
			rgt_batch = []
			slv_batch = []
			cat_batch = []
			temp_batch = []
			yd_batch = []
			rxn_id_batch = []
			for i in range(len(rxn_id)):
				rct_smiles = rxn_smiles[i].split('>>')[0]
				# print(rct_smiles)
				try:
					rct_mol = Chem.MolFromSmiles(rct_smiles)
					
				except:
					exception_count[0]+=1
					continue
				# [atom.ClearProp('molAtomMapNumber')for \
				# 	atom in rct_mol.GetAtoms() if atom.HasProp('molAtomMapNumber')]
				# rct_smiles = Chem.MolToSmiles(rct_mol)
				prd_smiles = rxn_smiles[i].split('>>')[1]
				try:
					prd_mol = Chem.MolFromSmiles(prd_smiles)

				except:
					exception_count[0]+=1
					continue

				if rct_mol is None or prd_mol is None:
					exception_count[0]+=1
					continue
				# print(prd_smiles)
				# [atom.ClearProp('molAtomMapNumber')for \
				# 	atom in prd_mol.GetAtoms() if atom.HasProp('molAtomMapNumber')]

				# prd_mol.UpdatePropertyCache()
				# Chem.SanitizeMol(prd_mol)
				# [a.SetProp('molAtomMapNumber', a.GetProp('old_molAtomMapNumber')) \
				# 	for (i, a) in enumerate(prd_mol.GetAtoms()) \
				# 	if 'old_molAtomMapNumber' in a.GetPropsAsDict()]
				# try:
				# 	rct_smiles = Chem.MolToSmiles(rct_mol)
				# 	prd_smiles = Chem.MolToSmiles(prd_mol)
				# except: 
				# 	continue
				# print(rct_smiles,prd_smiles)
				# try:
				# 	edits = summarize_reaction_outcome.summarize_reaction_outcome(rct_mol,prd_mol)
				# except:
				# 	# time.sleep(0.2)
				# 	exception_count[1]+=1
				# 	pass
				
				# if edits ==([],[],[],[]):
				# 	exception_count[2]+=1
				# 	pass
				# # print(edits)
				# edit_h_lost_vec = []
				# edit_h_gain_vec = []
				# edit_bond_lost_vec = []
				# edit_bond_gain_vec = []
				# try:
				# 	edit_h_lost_vec, edit_h_gain_vec, \
				# 		edit_bond_lost_vec, edit_bond_gain_vec = descriptors.edits_to_vectors(edits, rct_mol, atom_desc_dict = ATOM_DESC_DICT)
				# except KeyError as e: # sometimes molAtomMapNumber not found if hydrogens were explicit
				# 	exception_count[3]+=1
				# 	pass
				# for (e, edit_h_lost) in enumerate(edit_h_lost_vec):
				# 	if e >= N_e1: raise ValueError('N_e1 not large enough!')
				# 	x_h_lost[i, c, e, :] = edit_h_lost
				# for (e, edit_h_gain) in enumerate(edit_h_gain_vec):
				# 	if e >= N_e2: raise ValueError('N_e2 not large enough!')
				# 	x_h_gain[i, c, e, :] = edit_h_gain
				# for (e, edit_bond_lost) in enumerate(edit_bond_lost_vec):
				# 	if e >= N_e3: raise ValueError('N_e3 not large enough!')
				# 	x_bond_lost[i, c, e, :] = edit_bond_lost
				# for (e, edit_bond_gain) in enumerate(edit_bond_gain_vec):
				# 	if e >= N_e4: raise ValueRrror('N_e4 not large enough!')
				# 	x_bond_gain[i, c, e, :] = edit_bond_gain

				# x_h_lost[np.isnan(x_h_lost)] = 0.0
				# x_h_gain[np.isnan(x_h_gain)] = 0.0
				# x_bond_lost[np.isnan(x_bond_lost)] = 0.0
				# x_bond_gain[np.isnan(x_bond_gain)] = 0.0
				# x_h_lost[np.isinf(x_h_lost)] = 0.0
				# x_h_gain[np.isinf(x_h_gain)] = 0.0
				# x_bond_lost[np.isinf(x_bond_lost)] = 0.0
				# x_bond_gain[np.isinf(x_bond_gain)] = 0.0
				# print(len(edit_h_lost_vec),len(edit_h_gain_vec), len(edit_bond_lost_vec), len(edit_bond_gain_vec))
				# print(edit_h_lost_vec,edit_h_gain_vec, edit_bond_lost_vec, edit_bond_gain_vec)
				

				### also get product and reactant smiles
				try:
					[atom.ClearProp('molAtomMapNumber')for \
							atom in rct_mol.GetAtoms() if atom.HasProp('molAtomMapNumber')]
					[atom.ClearProp('molAtomMapNumber')for \
							atom in prd_mol.GetAtoms() if atom.HasProp('molAtomMapNumber')]
				except:
					exception_count[0]+=1
					continue
				try:
					rct_smiles = Chem.MolToSmiles(rct_mol,isomericSmiles=True)
					prd_smiles = Chem.MolToSmiles(prd_mol,isomericSmiles=True)
					print(rct_smiles,prd_smiles)
					# print(rct_smiles,prd_smiles)
					[pfp,rfp] = create_rxn_Morgan2FP(rct_smiles,prd_smiles)
				except:
					exception_count[0]+=1
					continue
				if pfp is None or rfp is None:
					exception_count[0]+=1
					continue
				if not (pfp.any() and rfp.any()):
					exception_count[0]+=1
					continue

				pfp_batch.append(pfp)
				rfp_batch.append(rfp)
				rgt_batch.append(rgt[i])
				slv_batch.append(slv[i])
				cat_batch.append(cat[i])
				temp_batch.append(temp[i])
				yd_batch.append(yd[i])
				rxn_id_batch.append(rxn_id[i])	
			pfp_mtx = sparse.csr_matrix(pfp_batch)
			rfp_mtx = sparse.csr_matrix(rfp_batch)

			

			# edit_vec = [edit_h_lost_vec, edit_h_gain_vec, \
			# 		edit_bond_lost_vec, edit_bond_gain_vec]
			while True:
				if results_queue.qsize()>100:
					time.sleep(0.1)
				else:
					break
			results_queue.put((pfp_mtx, rfp_mtx, rgt_batch, slv_batch, cat_batch, temp_batch, yd_batch, rxn_id_batch))


		except VanillaQueue.Empty:
			time.sleep(.1)
			pass

def coordinator(results_queue, rxn_id_list, edit_vec_list, pfp_mtx_list, rfp_mtx_list, rgt_list, slv_list, cat_list, temp_list, yd_list, doc_count, done):
	while True:
		try:
			(pfp_mtx, rfp_mtx, rgt, slv, cat, temp, yd, rxn_id) = results_queue.get()
			
			if rxn_id is None:
				done.value = 1
				print("coordinator saw done signal, stopping")
				break
			doc_count.value+=1
			# pfp_mtx_list.append([index,pfp_mtx])
			# pfp_label_list.append([index,pfp_label_chunk])
			# rfp_list.append([index,rfp_chunk])
			# rfp_label_list.append([index,rfp_label_chunk])
			# outcomes_list.append([index,outcomes_chunk])
			
			# rct_smiles_list.append([index,rct_smiles])
			# prd_smiles_list.append([index,prd_smiles])

			pfp_mtx_list.append(pfp_mtx)
			rfp_mtx_list.append(rfp_mtx)
			rgt_list.append(rgt)
			slv_list.append(slv)
			cat_list.append(cat)
			temp_list.append(temp)
			yd_list.append(yd)
			rxn_id_list.append(rxn_id)
			# edit_vec_list.append(edit_vec)
			# rct_smiles_list.append(rct_smiles)
			# prd_smiles_list.append(prd_smiles)

		except VanillaQueue.Empty:
			print('queue empty...')
			time.sleep(0.1)
			pass


###fetch data entries with one reagent or no reagent###
### data entry format:##
#   reaction edits, reagent

nb_workers = 8
manager = Manager()
doc_queue = Queue()
results_queue = Queue()
done = manager.list([0]*nb_workers)
coordinator_done = manager.Value('i',0)
rxn_id_list = manager.list([])
pfp_mtx_list = manager.list([])
rfp_mtx_list = manager.list([])
rgt_list = manager.list([])
slv_list = manager.list([])
cat_list = manager.list([])
temp_list = manager.list([])
yd_list = manager.list([])
edit_vec_list = manager.list([])
ATOM_DESC_DICT = manager.dict({})
exception_count = manager.list([0,0,0,0])
doc_count = manager.Value('i',0)

data_set_raw = []
rxn_id_list_raw = []
rxn_smiles_list_raw = []
rgt_list_raw = []
slv_list_raw = []
cat_list_raw = []
temp_list_raw = []
yd_list_raw =[]

MINIMUM_MAXPUB_YEAR = 1940
exception_ctr = {'no_cont':0,
				'exceed_2':0,
				'temp_out':0,
				'no_rxn':0,
				'no_smiles':0,
				'mul_prd':0,
				'multistep':0
				}
for doc in tqdm(instance_db.find({}, ['_id', 'RX_ID', 'RXD_RGTXRN','RXD_SOLXRN','RXD_CATXRN','RXD_NYD','RXD_T','RXD_STP'])):
	if doc['RXD_STP'] != ['1']:
                # print('skipped a multistep reaction')
		exception_ctr['multistep']+=1
		continue
	# try:
	temp = string_or_range_to_float(doc["RXD_T"])
	rgt = doc["RXD_RGTXRN"]
	slv = doc["RXD_SOLXRN"]
	cat = doc["RXD_CATXRN"]
	
	yd = doc["RXD_NYD"]
	if not rgt and not slv and not cat and (np.isnan(temp) or temp == -1):
		# print(rgt,slv,cat)
		# print(temp)
		# wait = raw_input("press anything to continue...")
		exception_ctr['no_cont']+=1
		continue
	if len(rgt)>=3 or len(slv)>= 3 or len(cat)>=2:
		exception_ctr['exceed_2']+=1
		continue


	# wait = raw_input("skipped the entry,press anything to continue...")
	if temp<-100 or temp >500:
		exception_ctr['temp_out']+=1
		continue
	# if example_doc['_id'] < skip_id: 
 #        continue
    # if data_generator.ctr < skip: continue 

    # Impose constraint on max publication year
    # if example_doc['RX_MAXPUB'] == -1: # now in query
    #     continue
    
	rxn_id = doc["RX_ID"]
	
	rxn_id = int(rxn_id[0])
	# print(rxn_id)
	# print(rxn_id)
	rxn = reaction_db.find_one({'_id':rxn_id}, ["RXN_SMILES",'RX_MAXPUB','RX_NUMREF'])
	if not rxn:
		exception_ctr['no_rxn']+=1 
		continue
	# if rxn['RX_MAXPUB'] == -1: continue
	# if (int(rxn['RX_MAXPUB'][0]) < MINIMUM_MAXPUB_YEAR) and (rxn['RX_NUMREF'] <= 1):
	# 	continue
	# print(rxn)
	try:
		rxn_smiles = rxn["RXN_SMILES"]
	except:
		exception_ctr['no_smiles']+=1
		continue
	if rxn_smiles == []:
		exception_ctr['no_smiles']+=1
		continue
	prd_smiles = rxn_smiles.split('>>')[1]
	if '.' in prd_smiles:
		exception_ctr['mul_prd']+=1
		continue

	# if rgt ==[]:
	# 	rgt = [-1]
	# if slv ==[]:
	# 	slv = [-1]
	# if cat ==[]:
	# 	cat = [-1]
	rgt.sort()
	slv.sort()
	cat.sort()

	rxn_id_list_raw.append(rxn_id)
	rxn_smiles_list_raw.append(rxn_smiles)
	rgt_list_raw.append(rgt)
	slv_list_raw.append(slv)
	cat_list_raw.append(cat)
	temp_list_raw.append(temp)
	yd_list_raw.append(yd)


print(len(rxn_id_list_raw))


###keep the instance with maximum yield
# df = pd.DataFrame(
# 	{'rxn_id':rxn_id_list_raw,
# 	 'rxn_smiles':rxn_smiles_list_raw,
# 	 'rgt':rgt_list_raw,
# 	 'yd':yd_list_raw
# 	})
# df_max_yd = df.sort_values('yd',ascending = False).groupby('rxn_id',as_index=False).first()
# rxn_id_list_raw = df_max_yd['rxn_id']
# rxn_smiles_list_raw = df_max_yd['rxn_smiles']
# rgt_list_raw = df_max_yd['rgt']
# yd_list_raw = df_max_yd['yd']
# print(len(rxn_id_list_raw))

workers = []
for wid in range(nb_workers):
	p = Process(target = processing_worker, args = (wid, doc_queue, results_queue, done,exception_count,))
	workers.append(p)
	p.start()
	print("worker {} started...".format(wid))

coordinator_p = Process(target = coordinator, args = (results_queue, rxn_id_list, edit_vec_list, pfp_mtx_list, rfp_mtx_list, rgt_list, slv_list, cat_list, temp_list, yd_list, doc_count, coordinator_done,))
coordinator_p.start()
print("coordinator started...")

batch_size = 1000
print(len(rxn_id_list_raw))
nb_batches = int(np.ceil(float(len(rxn_id_list_raw))/batch_size))
print(nb_batches)
max_index = len(rxn_id_list_raw)
start_index = 0
for i in tqdm(range(nb_batches)):
	end_index = min(start_index + batch_size,max_index)
	rxn_id = rxn_id_list_raw[start_index:end_index]
	rxn_smiles = rxn_smiles_list_raw[start_index:end_index]
	rgt = rgt_list_raw[start_index:end_index]
	slv = slv_list_raw[start_index:end_index]
	cat = cat_list_raw[start_index:end_index]
	temp = temp_list_raw[start_index:end_index]
	yd = yd_list_raw[start_index:end_index]
	while True:
		if doc_queue.qsize()>1000:
			time.sleep(0.1)
		else:
			break
	doc_queue.put((rxn_id, rxn_smiles, rgt, slv, cat, temp, yd))
	start_index += batch_size
 
for i in range(nb_workers):
	doc_queue.put((None,0,0,0,0,0,0))
print('loaded doc queue')

data_set_raw = []
rxn_id_list_raw = []
rxn_smiles_list_raw = []
rgt_list_raw = []
slv_list_raw = []
cat_list_raw = []
temp_list_raw = []
yd_list_raw =[]
# time.sleep(5)
# for p in workers:
# 	p.join()
import time
starttime=time.time()

while True:
	# print(doc_count.value)
	if(all(done)):
		for p in workers:
			p.terminate()
			time.sleep(0.1)
			# print(p.is_alive())
	if all([p.is_alive() == False for p in workers]):
		print("all workers done, continue to next step...")
		results_queue.put((0,0,0,0,0,0,0,None))
		print("added poison pill for coordinator")
		print([p.is_alive() == False for p in workers])
		break
	else:
		time.sleep(0.1)
		pass

while True:
	if coordinator_done.value == 1:
		coordinator_p.terminate()
		time.sleep(0.1)
		print(coordinator_p.is_alive())
		print("coordinator done, continue to next step...")
		break
	else:
		time.sleep(0.1)
		pass

print(datetime.datetime.now())
	# except:
	# 	continue
print(exception_count)
print(doc_count.value)
print(len(rxn_id_list),len(pfp_mtx_list),len(rfp_mtx_list),len(rgt_list),len(slv_list),len(cat_list), len(temp_list), len(yd_list))

index_list = range(len(rxn_id_list))

# flat_list = [item for sublist in l for item in sublist]

rxn_id_list = list(itertools.chain.from_iterable(rxn_id_list))
# edit_vec_list = [edit_vec_list[i] for i in index_list]
pfp_csrmtx = sparse.vstack(pfp_mtx_list)
rfp_csrmtx = sparse.vstack(rfp_mtx_list)

rgt_list = list(itertools.chain.from_iterable(rgt_list))
slv_list = list(itertools.chain.from_iterable(slv_list))
cat_list = list(itertools.chain.from_iterable(cat_list))
temp_list = list(itertools.chain.from_iterable(temp_list))
yd_list = list(itertools.chain.from_iterable(yd_list))
print(len(rgt_list),len(slv_list))
rgt_min_count = 100
slv_min_count = 100
cat_min_count = 100

####flatten rgt slv cat lists

print(cat_list[:10])
print(rgt_list[:10])
flat_rgt_list = list(itertools.chain.from_iterable(rgt_list))
flat_slv_list = list(itertools.chain.from_iterable(slv_list))
flat_cat_list = list(itertools.chain.from_iterable(cat_list))

#### count frequency
rgt_counter = Counter(flat_rgt_list)
slv_counter = Counter(flat_slv_list)
cat_counter = Counter(flat_cat_list)
# 

list_of_metal_atoms = ['Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag',\
							'Cd','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Rf','Db','Sg','Bh','Hs','Mt','Ds','Rg',
							# 'Cn',
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
							# 'Copernicium',
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
list_of_metals = [metal.lower() for metal in list_of_full_metal_names]+list_of_metal_atoms
cat_name_list = []
cat_smiles_list = []
cat_count_list = []
metal_cat_dict = {}
metal_cat_list = []
for key, value in cat_counter.iteritems():
	metal_cat_dict[key] = False
	cat = ''
	cat_smi = ''
	try:
		cat = chemical_db.find_one({'_id':key})['IDE_CN']
	except:
		# print('chemical not found!\n')
		pass
	# try:
	# 	cat_smi = chemical_db.find_one({'_id':key})['SMILES']
	# except:
	# 	pass

	if any(metal in cat for metal in list_of_metals):
		metal_cat_dict[key] = True
	
	cat_name_list.append(cat)
	# cat_smiles_list.append(cat_smi)
	cat_count_list.append(value)
	metal_cat_list.append(metal_cat_dict[key])

cat_df = pd.DataFrame({'cat':cat_name_list,'count':cat_count_list, 'metal':metal_cat_list})
cat_df.to_csv('catalyst_count.csv')

rgt_name_list = []
rgt_count_list = []
for key, value in rgt_counter.iteritems():
	try:
		rgt = chemical_db.find_one({'_id':key})['IDE_CN']
	except:
		# print('chemical not found!\n')
		continue
	rgt_name_list.append(rgt)
	rgt_count_list.append(value)

rgt_df = pd.DataFrame({'rgt':rgt_name_list,'count':rgt_count_list})
rgt_df.to_csv('reagent_count.csv')
### filter by frequency
# rare_rgt_set = {rgt for rgt in rgt_counter if rgt_counter[rgt] < rgt_min_count}
# rare_slv_set = {slv for slv in slv_counter if slv_counter[slv] < slv_min_count}
metal_cat_set = {cat for cat in cat_counter if metal_cat_dict[cat]}


flat_rgt_set = set(flat_rgt_list)
flat_slv_set = set(flat_slv_list)
flat_cat_set = set(flat_cat_list)

#### reagents that are not used as catalysts
# non_cat_rgt_set = flat_rgt_set - metal_cat_set


# rgt_cat_list = [rgt_list[i]+cat_list[i] for i in range(len(rgt_list))]

cat_list = [cat_list[i]+list(set(rgt_list[i]).intersection(metal_cat_set)) for i in range(len(cat_list))]
rgt_list = [list(set(rgt_list[i]) - set(rgt_list[i]).intersection(metal_cat_set)) for i in range(len(rgt_list))]

cat_list = [item if item!=[] else [-1] for item in cat_list]
slv_list = [item if item!=[] else [-1] for item in slv_list]
rgt_list = [item if item!=[] else [-1] for item in rgt_list]

flat_rgt_list = list(itertools.chain.from_iterable(rgt_list))
flat_slv_list = list(itertools.chain.from_iterable(slv_list))
flat_cat_list = list(itertools.chain.from_iterable(cat_list))

#### count frequency
rgt_counter = Counter(flat_rgt_list)
slv_counter = Counter(flat_slv_list)
cat_counter = Counter(flat_cat_list)

### filter by frequency
rare_rgt_set = {rgt for rgt in rgt_counter if rgt_counter[rgt] < rgt_min_count}
rare_slv_set = {slv for slv in slv_counter if slv_counter[slv] < slv_min_count}
rare_cat_set = {cat for cat in cat_counter if cat_counter[cat] < cat_min_count}

print([[chemical_db.find_one({'_id':key})['IDE_CN'] for key in item if key!=-1] for item in cat_list[:10]])
print([[chemical_db.find_one({'_id':key})['IDE_CN'] for key in item if key!=-1] for item in rgt_list[:10]])
# print(rare_rgt_set)
# print(rare_cat_set)

### remove rare cases
rgt_list = [item if not(set(item).intersection(rare_rgt_set)) else [] for item in rgt_list]
slv_list = [item if not(set(item).intersection(rare_slv_set)) else [] for item in slv_list]
cat_list = [item if not(set(item).intersection(rare_cat_set)) else [] for item in cat_list]
# final_rgt_set = flat_rgt_set-rare_rgt_set- flat_cat_set
num_rgt = len(rgt_counter)- len(rare_rgt_set)
print("total number of unique rgt is {}, truncated number of rgt is {}".format(len(rgt_counter),num_rgt))
num_slv = len(slv_counter)- len(rare_slv_set)
print("total number of unique slv is {}, truncated number of slv is {}".format(len(slv_counter),num_slv))
num_cat = len(cat_counter)- len(rare_cat_set)
print("total number of unique cat is {}, truncated number of cat is {}".format(len(cat_counter),num_cat))



indices_to_delete = [] 
for i in range(len(rgt_list)):
	if rgt_list[i] == [] or slv_list[i] == [] or cat_list[i] == []:
		indices_to_delete.append(i)
index_list = range(len(rgt_list))
indices_to_keep = list(set(index_list) - set(indices_to_delete))

# print(indices_to_delete)
### remove reactions with rare context
# rxn_id_list = [i for j,i in tqdm(enumerate(rxn_id_list)) if j not in indices_to_delete]
# pfp_mtx_list = [i for j,i in tqdm(enumerate(pfp_mtx_list)) if j not in indices_to_delete]
# rfp_mtx_list = [i for j,i in tqdm(enumerate(rfp_mtx_list)) if j not in indices_to_delete]
pfp_csrmtx = pfp_csrmtx[indices_to_keep,:]
rfp_csrmtx = rfp_csrmtx[indices_to_keep,:]

rxn_id_list = [rxn_id_list[i] for i in tqdm(indices_to_keep)]
rgt_list = [rgt_list[i] for i in tqdm(indices_to_keep)]
slv_list = [slv_list[i] for i in tqdm(indices_to_keep)]
cat_list = [cat_list[i] for i in tqdm(indices_to_keep)]
temp_list = [temp_list[i] for i in tqdm(indices_to_keep)]
yd_list = [yd_list[i] for i in tqdm(indices_to_keep)]

# rgt_list = [i for j,i in tqdm(enumerate(rgt_list)) if j not in indices_to_delete]
# slv_list = [i for j,i in tqdm(enumerate(slv_list)) if j not in indices_to_delete]
# cat_list = [i for j,i in tqdm(enumerate(cat_list)) if j not in indices_to_delete]
# temp_list = [i for j,i in tqdm(enumerate(temp_list)) if j not in indices_to_delete]
# yd_list = [i for j,i in tqdm(enumerate(yd_list)) if j not in indices_to_delete]
print("records after truncation:{}".format(len(rgt_list)))
print("finished truncating...")
print(datetime.datetime.now())

#####OneHot Encoding for reagent solvent and catalyst################################
rgt_1_list = [rgt[0] if len(rgt)>=1 else -1 for rgt in rgt_list]
rgt_2_list = [rgt[1] if len(rgt)>=2 else -1 for rgt in rgt_list]
slv_1_list = [slv[0] if len(slv)>=1 else -1 for slv in slv_list]
slv_2_list = [slv[1] if len(slv)>=2 else -1 for slv in slv_list]
cat_1_list = [cat[0] if len(cat)>=1 else -1 for cat in cat_list]

# print(flat_rgt_list)
# print(rgt_1_list)
# print(rgt_2_list)
# print(rgt_list)

# context_list = ['.'.join([','.join(str(e) for e in sorted(rgt)),','.join(str(e) for e in sorted(slv)),','.join(str(e) for e in sorted(cat))]) for rgt,slv,cat in zip(rgt_list, slv_list, cat_list)]
rgt_le = LabelEncoder()
slv_le = LabelEncoder()
cat_le = LabelEncoder()

flat_rgt_list = list(set(flat_rgt_list))
flat_slv_list = list(set(flat_slv_list))
flat_cat_list = list(set(flat_cat_list))

flat_rgt_list.append(-1)
flat_slv_list.append(-1)
flat_cat_list.append(-1)

rgt_le.fit(flat_rgt_list)
slv_le.fit(flat_slv_list)
cat_le.fit(flat_cat_list)


# context_label_list = le.fit_transform(context_list)
rgt_1_label_list = rgt_le.transform(rgt_1_list)
rgt_2_label_list = rgt_le.transform(rgt_2_list)
slv_1_label_list = slv_le.transform(slv_1_list)
slv_2_label_list = slv_le.transform(slv_2_list)
cat_1_label_list = cat_le.transform(cat_1_list)

r1_ohe = OneHotEncoder()
r2_ohe = OneHotEncoder()
s1_ohe = OneHotEncoder()
s2_ohe = OneHotEncoder()
c1_ohe = OneHotEncoder()

# context_onehot_sparse = ohe.fit_transform(context_label_list.reshape(-1,1))
rgt_1_onehot_sparse = r1_ohe.fit_transform(rgt_1_label_list.reshape(-1,1))
rgt_2_onehot_sparse = r2_ohe.fit_transform(rgt_2_label_list.reshape(-1,1))
slv_1_onehot_sparse = s1_ohe.fit_transform(slv_1_label_list.reshape(-1,1))
slv_2_onehot_sparse = s2_ohe.fit_transform(slv_2_label_list.reshape(-1,1))
cat_1_onehot_sparse = c1_ohe.fit_transform(cat_1_label_list.reshape(-1,1))

print("the shapes of the sparse matrices are (rgt 1 rgt 2 slv 1 slv 2 cat 1")
print(rgt_1_onehot_sparse.shape)
print(rgt_2_onehot_sparse.shape)
# print(type(rgt_onehot_sparse))
print(slv_1_onehot_sparse.shape)
print(cat_1_onehot_sparse.shape)
print(len(temp_list))

# context_onehot_sparse = sparse.hstack([cat_1_onehot_sparse,rgt_1_onehot_sparse,rgt_2_onehot_sparse,slv_1_onehot_sparse,slv_2_onehot_sparse],format = 'csr')
print("finished onehot encoding...")
print(datetime.datetime.now())
# print("the shape of the context sparse matrix")
# print(context_onehot_sparse.shape)
# print(rxn_id_list[:20])
# print(context_onehot_sparse[10,:].todense())


# df1 = pd.DataFrame({'rxn_id':rxn_id_list,
# 					'context':list(context_onehot_sparse)})
# df1_grouped = df1.groupby('rxn_id',as_index = False).sum()
# df2 = pd.DataFrame({'rxn_id':rxn_id_list,
# 					'pfp': pfp_mtx_list,
# 					'rfp': rfp_mtx_list
# 					})
# df2_grouped = df2.groupby('rxn_id',as_index = False).first()

# rxn_id_list = df1_grouped['rxn_id']
# context_onehot_list = list(context_onehot_sparse)
# rgt_1_onehot_list = list(rgt_1_onehot_sparse)
# rgt_2_onehot_list = list(rgt_2_onehot_sparse)
# slv_1_onehot_list = list(slv_1_onehot_sparse)
# slv_2_onehot_list = list(slv_2_onehot_sparse)
# cat_1_onehot_list = list(cat_1_onehot_sparse)
# pfp_mtx_list = df2_grouped['pfp']
# rfp_mtx_list = df2_grouped['rfp']
##shuffle

# original_rxn_id_file = "/home/hanyug/Make-It/makeit/context_pred/preprocessed_data/separate/fullset_rearrange_test/rxn_ids.pickle"
# with open(original_rxn_id_file,"r") as RID_O:
# 	original_rxn_id_list=pickle.load(RID_O)

# seen = set()
# unique_id_list = [x for x in original_rxn_id_list if not (x in seen or seen.add(x))]


index_list = range(len(rxn_id_list))
groups = [list(g) for _,g in itertools.groupby(sorted(zip(rxn_id_list,index_list)),operator.itemgetter(0))]
random.shuffle(groups)
# print(len(groups))
# grouped_rid_list = [groups[i][0][0] for i in range(len(groups))]
# print(len(grouped_rid_list))
# order_list = [grouped_rid_list.index(rid) for rid in unique_id_list]
# print(len(order_list))
# groups = [groups[i] for i in order_list]
shuffled = [item for group in groups for item in group]
[rxn_id_list, index_list] = zip(*shuffled)
# rxn_id_list = [rxn_id_list[i] for i in index_list]
# edit_vec_list = [edit_vec_list[i] for i in index_list]
pfp_csrmtx = pfp_csrmtx[index_list,:]
rfp_csrmtx = rfp_csrmtx[index_list,:]
# context_onehot_list = [context_onehot_list[i] for i in index_list]
rgt_1_onehot_sparse = rgt_1_onehot_sparse[index_list,:]
rgt_2_onehot_sparse = rgt_2_onehot_sparse[index_list,:]
slv_1_onehot_sparse = slv_1_onehot_sparse[index_list,:]
slv_2_onehot_sparse = slv_2_onehot_sparse[index_list,:]
cat_1_onehot_sparse = cat_1_onehot_sparse[index_list,:]
temp_list = [temp_list[i] for i in index_list]
yd_list = [yd_list[i] for i in index_list]

print("finished random shuffling...")
print(len(set(rxn_id_list)))
# print(groups)
print(datetime.datetime.now())

# context_csrmtx = sparse.vstack(context_onehot_list)
print(len(rxn_id_list))
print(pfp_csrmtx.shape)
# print(rxn_id_list[:10])
# print(rxn_id_list[10],rgt_list[10], slv_list[10], cat_list[10], yd_list[10])
# print(context_label_list[10], max(context_label_list),min(context_label_list))
# # print(edit_vec_list[10])
# print(pfp_csrmtx[10,:].indices,rfp_csrmtx[10,:].indices)
# print(context_csrmtx[10,:].todense())
print(np.min(np.sum(rgt_1_onehot_sparse, axis =1)))
print(np.max(np.sum(rgt_1_onehot_sparse, axis =1)))
print(exception_count[0])
print(exception_ctr)
# print(context_onehot_list[10][0,0:24].indices)
# print(cat_le.inverse_transform(context_onehot_list[10][0,0:cat_1_onehot_sparse.shape[1]].indices))
# data_set = list(set(data_set))
# print(len(data_set))
rxn_id_file = ###output file path
# edit_vec_file = "/home/hanyug/Make-It/makeit/context_pred/preprocessed_data/edit_vecs.pickle"
rgt_file = ###output file path
slv_file =  ###output file path
cat_file =  ###output file path
temp_file =  ###output file path
yd_file =  ###output file path
# context_file = "/home/hanyug/Make-It/makeit/context_pred/preprocessed_data/separate/context.pickle"
rgt_1_mtx_file =  ###output file path
rgt_2_mtx_file =  ###output file path
slv_1_mtx_file =  ###output file path
slv_2_mtx_file =  ###output file path
cat_1_mtx_file =  ###output file path
# temp_mtx_file = "/home/hanyug/Make-It/makeit/context_pred/preprocessed_data/separate/full1/temp_mtx"
# context_mtx_file = "/home/hanyug/Make-It/makeit/context_pred/preprocessed_data/separate/full1/context_mtx"
pfp_mtx_file =  ###output file path
rfp_mtx_file =  ###output file path

##encoders
rgt_le_file =  ###output file path
slv_le_file =  ###output file path
cat_le_file =  ###output file path

r1_ohe_file =  ###output file path
r2_ohe_file =  ###output file path
s1_ohe_file =  ###output file path
s2_ohe_file =  ###output file path
c1_ohe_file =  ###output file path


sparse.save_npz(rgt_1_mtx_file,rgt_1_onehot_sparse)
sparse.save_npz(rgt_2_mtx_file,rgt_2_onehot_sparse)
sparse.save_npz(slv_1_mtx_file,slv_1_onehot_sparse)
sparse.save_npz(slv_2_mtx_file,slv_2_onehot_sparse)
sparse.save_npz(cat_1_mtx_file,cat_1_onehot_sparse)

sparse.save_npz(pfp_mtx_file,pfp_csrmtx)
sparse.save_npz(rfp_mtx_file,rfp_csrmtx)
# sparse.save_npz(context_mtx_file,context_csrmtx)
with open(rxn_id_file,"w") as RID:
	pickle.dump(rxn_id_list,RID)


# with open(edit_vec_file,"w") as EDT:
# 	pickle.dump(edit_vec_list,EDT)
with open(rgt_file,"w") as RGT_F:
	pickle.dump(rgt_counter,RGT_F)
with open(slv_file,"w") as SLV_F:
	pickle.dump(slv_counter,SLV_F)
with open(cat_file,"w") as CAT_F:
	pickle.dump(cat_counter,CAT_F)
with open(temp_file,"w") as T_F:
	pickle.dump(temp_list,T_F)
with open(yd_file,"w") as YD_F:
	pickle.dump(yd_list,YD_F)

with open(rgt_le_file,"w") as RLE_F:
	pickle.dump(rgt_le, RLE_F)
with open(slv_le_file,"w") as SLE_F:
	pickle.dump(slv_le, SLE_F)
with open(cat_le_file,"w") as CLE_F:
	pickle.dump(cat_le, CLE_F)

with open(r1_ohe_file,"w") as R1OE_F:
	pickle.dump(r1_ohe, R1OE_F)
with open(r2_ohe_file,"w") as R2OE_F:
	pickle.dump(r2_ohe, R2OE_F)
with open(s1_ohe_file,"w") as S1OE_F:
	pickle.dump(s1_ohe, S1OE_F)
with open(s2_ohe_file,"w") as S2OE_F:
	pickle.dump(s2_ohe, S2OE_F)
with open(c1_ohe_file,"w") as C1OE_F:
	pickle.dump(c1_ohe, C1OE_F)

# with open(rgt_label_file,"w") as RGT_L_F:
# 	pickle.dump(rgt_label_list,RGT_L_F)
# with open(slv_label_file,"w") as SLV_L_F:
# 	pickle.dump(slv_label_list,SLV_L_F)
# with open(cat_label_file,"w") as CAT_L_F:
# 	pickle.dump(cat_label_list,CAT_L_F)
# label_file = "/home/hanyu/rxn_ids.pickle"

# with open(data_file,"wb") as f1:
# 	pickle.dump(data_set, f1, -1)
