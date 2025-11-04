# To apply mann whitney u test from tsfresh to filter out data features > FDR 0.05 on the individual tissue specific eQTL effect table with 4366 individual samples
# Use GridSearchCV to search the optimized hyperparameter for the PD logistic predictor model
# Sklearn Machine Learning algorithms are used to create PD predictive models from the individual tissue specific eQTL effect table
# Inputs: Weighted_eQTL_matrix.txt


# Outputs: 
# PD_GWAS_Nalls_finalfull_Mann0.05_lg_saga_elect0.6C0.5max800_beta_20122020weights.txt : the model weights of PD model-1 
# PD_GWAS_Nalls_finalfull_Mann0.05_lg_saga_elect0.6C0.5max800_beta_20122020.sav : PD model-1
# PD_GWAS_Nalls_finalfull_Mann0.05_lg_saga_elect0.6C0.5max800_beta_20122020features.txt : The model component information of PD model-1
# PD_GWAS_Nalls_finalfull_Mann0.05_lg_saga_elect0.6C0.5max800_beta_20122020_results.txt : PD model-1 performance results

import numpy as np
import pandas as pd


from tsfresh import extract_features, select_features
from  tsfresh.feature_selection.relevance import calculate_relevance_table

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn import linear_model

from sklearn.model_selection import GridSearchCV
eQTL_table = pd.read_csv("data/Weighted_eQTL_matrix.txt", sep="\t")
x_features = eQTL_table[eQTL_table.columns[6:]]
y_phenotype = eQTL_table['PHENOTYPE'] - 1

#mann whitney u test from tsfresh to filter out data features > FDR 0.05
X_selected0_1 = select_features(x_features, y_phenotype,test_for_binary_target_real_feature='mann', test_for_real_target_binary_feature='mann', fdr_level=0.05 )

full_columns = list(eQTL_table.columns[:6]) + list( X_selected0_1.columns)
selected_std_full = eQTL_table[full_columns]

X = X_selected0_1
Y = eQTL_table['PHENOTYPE'] - 1

#search space
parameters = {'C':[0.01,0.05,0.1,0.5,1,10,20,30],'max_iter':[ 200,500,800,1000,1200,1400,1500,1600],'l1_ratio':[1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]}
lg_clf = LogisticRegression(random_state=1, solver='saga',n_jobs=-1, penalty='elasticnet' )
grid_clf = GridSearchCV(lg_clf, parameters, scoring='roc_auc', n_jobs=-1, cv=10)
grid_clf.fit(X,Y)

f= open("data/PD_GWAS_Nalls_finalfull_Mann0.05_lg_saga_elect0.6C0.5max800_beta_20122020_results.txt","w+")
# PD_GWAS_Nalls_finalfull_Mann0.05_lg_saga_elect0.6C0.5max800_beta_20122020_results.txt contains the predictor performance information

f.write('PD_GWAS_Nalls_finalfull_Mann0.05_lg_saga_elect0.6C0.5max800_beta_20122020\n')
f.write('grid_clf.best_score_: ' + str(grid_clf.best_score_) + '\n')
f.write('grid_clf.best_estimator_: ' + str(grid_clf.best_estimator_) + '\n')
f.write('grid_clf.best_params_: ' + str(grid_clf.best_params_) + '\n')
f.write('grid_clf.scorer_: ' + str(grid_clf.scorer_) + '\n')
f.write('grid_clf.best_score_: ' + str(grid_clf.best_score_) + '\n')


lg_clf_best_grid = grid_clf.best_estimator_

model_Max = roc_auc_score(Y, lg_clf_best_grid.predict_proba(X)[:,1])
test_score = roc_auc_score(y_test, lg_clf_best_grid.predict_proba(x_test)[:,1])
num_coef = np.sum(lg_clf_best_grid.coef_[0,:] != 0)

cv_results = grid_clf.cv_results_
std_cv = cv_results['std_test_score'][grid_clf.best_index_]

f.write('--------------------------------------------------------------------\n\n')
f.write('In-Sample AUC: ' + str(model_Max) + '\n')
f.write('MeanCV AUC: ' + str(grid_clf.best_score_) + '\n')
f.write('Standard Deviation CV AUC: ' + str(std_cv) + '\n')
f.write('Test sample AUC: ' + str(test_score) + '\n')
f.write('num_coef: ' + str(num_coef) + '\n')
f.close()

X_header = np.array(X.columns)
best_clf =  grid_clf.best_estimator_
data_array = np.vstack((X_header,best_clf.coef_[0,:]))
model_weights = pd.DataFrame(data=data_array.T,columns=['Data_feature', 'Weight'])
model_weights.to_csv('data/PD_GWAS_Nalls_finalfull_Mann0.05_lg_saga_elect0.6C0.5max800_beta_20122020weights.txt', sep='\t',index=False,line_terminator='\n')
# PD_GWAS_Nalls_finalfull_Mann0.05_lg_saga_elect0.6C0.5max800_beta_20122020weights.txt : the model weights of PD model-1 

import pickle
filename = 'data/PD_GWAS_Nalls_finalfull_Mann0.05_lg_saga_elect0.6C0.5max800_beta_20122020.sav'
pickle.dump(lg_clf_best_grid, open(filename, 'wb'))
# PD_GWAS_Nalls_finalfull_Mann0.05_lg_saga_elect0.6C0.5max800_beta_20122020.sav : PD model-1

model_com = model_weights.loc[model_weights['Weight'] != 0 ]
model_com['Data_feature'].to_csv('data/PD_GWAS_Nalls_finalfull_Mann0.05_lg_saga_elect0.6C0.5max800_beta_20122020features.txt', sep='\t',header=False,index=False,line_terminator='\n')
# PD_GWAS_Nalls_finalfull_Mann0.05_lg_saga_elect0.6C0.5max800_beta_20122020features.txt : The model component information of PD model-1
