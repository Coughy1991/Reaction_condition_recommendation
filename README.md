# Reaction condition recommendation

This is the code for data preprocessing, model training and evaluation described in the paper "Using Machine Learning to Predict Suitable Conditions for Organic Reactions".

## Dependencies
Data preprocessing step uses RDKit to calculate fingerprints from SMILES of the reactants and products, and uses sklearn for onehot encoding of the catalysts, solvents and reagents. The neural network model is built using Keras

## Data preparation
The code for data preprocessing is prepare_data_cont_2_rgt_2_slv_1_cat_temp_deploy.py. Our data source is Reaxys, which cannot be disclosed, but the principle of data processing can be transferred to other datasets that includes the following key steps:
Calculating the fingerprints of the reactants and the products
Counting the frequencies of catalysts, solvents and reagents and truncate based on frequency
Creating one hot vectors for catalysts, solvents and reagents

## Model building and training
train_model_c_s_r_deploy.py includes model building and training. As described in the paper, the model takes a hierarchical structure and predicts up to one catalyst, two solvents and two reagents and temperature of a reaction. The trained model is stored separately at https://figshare.com/s/e792359b2ce5e1c1a31f, since the file is too large for github.

## Testing with trained model
neuralnetwork.py is a script that uses the trained model to predict conditions for given organic reactions. It defines a NeuralNetContextRecommender class that can predict the conditions given the SMILES of the reactants and product. A user friendly version of the model is available at http://askcos.mit.edu/context.

results_analysis_with_null_model_deploy.py is the script used for generating the quantitative statistics in the paper.
