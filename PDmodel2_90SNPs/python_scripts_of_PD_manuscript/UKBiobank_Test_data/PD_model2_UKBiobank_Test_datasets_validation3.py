# Author: Daniel Ho 14/09/2021
#
# This python script is to validate the PD model-2 predictor model with the 30 sets of Test data derived from UK Biobank Parkinson's disease data
#
# Input datasets:
# After_0.05_Mann_selected_header.txt (The Mann Whitney (FDR = 0.5) selected header columns of the PD tissue-specific eQTL matrix derived from WTCCC)
# PD_UKBio_full_NallsGWAS_a_41SNP_cases_beta_928_eQTL_table.txt (PD case Test data derived from UK Biobank Parkinson's disease data)
# PD_Nalls_90_58Conly_full_beta_lg_C0.6elast0.1max130_04122020features.txt (Feature columes of PD model-2)
# PD_Nalls_90_58Conly_full_beta_lg_C0.6elast0.1max130_04122020.sav (Saved PD model-2)
# UKBio_testdata1.txt to UKBio_testdata30.txt (PD control Test datasets derived from UK Biobank Parkinson's disease data)
# 
# Output data:
# 30_UKBio_PD_AUCs.txt (The result file contains the 30 AUC validation results) 

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn import linear_model
import pandas as pd
from sklearn.model_selection import GridSearchCV
from tsfresh import extract_features, select_features
from  tsfresh.feature_selection.relevance import calculate_relevance_table
import pickle
import sklearn.metrics as metrics

org_table = pd.read_csv("data/After_0.05_Mann_selected_header.txt", sep="\t")

features = pd.read_csv("PD_Nalls_90_58Conly_full_beta_lg_C0.6elast0.1max130_04122020features.txt", names = ['fields'], sep="\t")
model_com = features['fields'].values.tolist()

filename = 'PD_Nalls_90_58Conly_full_beta_lg_C0.6elast0.1max130_04122020.sav'
loaded_model = pickle.load(open(filename, 'rb'))

lg_clf = loaded_model


num = [1,2,3,4,5,6,7,8,9,10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]

AUCs = []

for p in num:
	test_data = pd.read_csv('data/UKBio_testdata' + str(p) + '.txt', sep="\t")
	y_test = test_data['PHENOTYPE'] - 1
	x_test = test_data[test_data.columns[6:]]
	for fe in org_table.columns[6:]:
		if fe not in test_data.columns[6:]:
			x_test[fe] = 0
	x_test2 = x_test[org_table.columns[6:]]
	AUCs.append(roc_auc_score(y_test, lg_clf.predict_proba(x_test2)[:,1]))
	
results = pd.DataFrame(AUCs,columns=['UKBIO PD AUC'])
results.to_csv("30_UKBio_PD_AUCs.txt", sep='\t', index=False)


