# Use RepeatedKFold to do 5 x 10 fold validation of PD model-1 regularized logistic regression model regression model

# Inputs: Weighted_eQTL_matrix.txt
# The full tissue specific eQTL table

# Outputs: 
# 10fold_PDModel-1_hyperparameteres_model{count}weights.txt where count = 1 to 50 contains the model components and their weights for each randomized model
# AUC_results_of_50_regularized_predictors_with_PDmodel-1_hyperparameteres.txt contains the AUC results for the 50 randomized predictors
# Preformance_of_50_regularized_predictors_with_PDmodel-1_hyperparameteres.txt contains the performance results for the 50 randomized predictors


import statistics as st
import numpy as np
import pandas as pd
import statistics as st
from tsfresh import extract_features, select_features
from  tsfresh.feature_selection.relevance import calculate_relevance_table
from tsfresh.transformers import  FeatureSelector


import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn import linear_model
from sklearn.model_selection import RepeatedKFold

from sklearn.model_selection import GridSearchCV

eQTL_table = pd.read_table('data/Weighted_eQTL_matrix.txt')

Y_pd = eQTL_table['PHENOTYPE'] - 1

X_pd = eQTL_table[eQTL_table.columns[6:]]

X = X_pd
Y = Y_pd

AUC_result = list() # This list will store your results
selected_features = list()
classifiers = list()
# Initialize tsfresh feature selector
# Initialize 10 times repeated 5-fold cross validation
rkf = RepeatedKFold(n_splits=10, n_repeats=5, random_state=1)
# This loop will iterate 50 times


clf = LogisticRegression(C=0.5,max_iter=800, l1_ratio=0.6, random_state=1, solver='saga',n_jobs=-1, penalty='elasticnet')
select = FeatureSelector(fdr_level=0.05)

count = 0
for train_index, test_index in rkf.split(X):
# print("TRAIN:", train_index, "TEST:", test_index)
	X_train, X_test = X.iloc[train_index], X.iloc[test_index]
	y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
# 2. Apply Mann_Whitley test on the 90% training dataset
	
	X_train_sel= select.fit_transform(X_train, y_train)
# 3. Fit predictor
	
	clf.fit(X_train_sel, y_train)
# 4. Validate predictor
# 

	X_test_sel = select.transform(X_test)
	y_pred = clf.predict_proba(X_test_sel)[:,1]
	auc = roc_auc_score(y_test, y_pred)
# 5. Store result
	AUC_result.append(auc)
	selected_features.append(select.relevant_features)
	classifiers.append(clf)

	X_header = np.array(X_train_sel.columns)
#        best_clf = clf
	best_clf = clf
	count = count + 1
	data_array = np.vstack((X_header,best_clf.coef_[0,:]))
	model_weights = pd.DataFrame(data=data_array.T,columns=['Data_feature', 'Weight'])
	m_name = 'data/10fold_PDModel-1_hyperparameteres_model' + str(count) + 'weights.txt'
	model_weights.to_csv(m_name, sep='\t',index=False, line_terminator='\n')


# Now do an in-sample evaluation
# 6. Apply Mann-Whitley test on the 100% original eQTL dataset.
X_sel = select.fit_transform(X, Y)
# 7. Fit predictor to statistically significant features (just once!!!)
clf.fit(X_sel, Y)
y_pred = clf.predict_proba(X_sel)[:,1]
# This in-sample AUC should be better than your the AUCs from your repeated cross-validation
auc = roc_auc_score(Y, y_pred)

#AUC results from the 50 predictors 
AUC_out = pd.DataFrame(AUC_result, columns=['AUC'])
AUC_out.to_csv("data/AUC_results_of_50_regularized_predictors_with_PDmodel-1_hyperparameteres.txt", sep='\t',index=False, line_terminator='\n')

AUC_std= st.stdev(AUC_result)
AUC_mean= st.mean(AUC_result)

num_coef = np.sum(clf.coef_[0,:] != 0)

#Performance results for the randomized predictors
f= open("data/Preformance_of_50_regularized_predictors_with_PDmodel-1_hyperparameteres.txt","w+")

f.write('50_regularized_predictors_with_PDmodel-1_hyperparameteres\n')

f.write('In-Sample AUC: ' + str(auc) + '\n')
f.write('MeanCV AUC: ' + str(AUC_mean) + '\n')
f.write('Standard Deviation CV AUC: ' + str(AUC_std) + '\n')

f.write('num_coef: ' + str(num_coef) + '\n')

f.close()








