# To apply mann whitney u test from tsfresh to filter out data features > FDR 0.05 on the individual tissue specific eQTL effect table with 4366 individual samples
# Use GridSearchCV to search the optimized hyperparameter for the PD logistic predictor model
# Sklearn Machine Learning algorithms are used to create PD predictive models from the individual tissue specific eQTL effect table
# Inputs: Weighted_eQTL_matrix.txt


# Outputs: 
# PD_Nalls_90_58Conly_full_beta_lg_C0.6elast0.1max130_04122020weights.txt : the model weights of PD model-2
# PD_Nalls_90_58Conly_full_beta_lg_C0.6elast0.1max130_04122020.sav : PD model-2
# PD_Nalls_90_58Conly_full_beta_lg_C0.6elast0.1max130_04122020features.txt : The model component information of PD model-2
# PD_Nalls_90_58Conly_full_beta_lg_C0.6elast0.1max130_04122020_results.txt : PD model-2 performance results

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

eQTL_table = pd.read_csv("data/Weighted_eQTL_matrix.txt", sep="\t")
x_features = eQTL_table[eQTL_table.columns[6:]]
y_phenotype = eQTL_table['PHENOTYPE'] - 1

X_selected0_1 = select_features(x_features, y_phenotype,test_for_binary_target_real_feature='mann', test_for_real_target_binary_feature='mann', fdr_level=0.05 )

full_columns = list(eQTL_table.columns[:6]) + list( X_selected0_1.columns)
selected_std_full = eQTL_table[full_columns]

X = X_selected0_1
Y = eQTL_table['PHENOTYPE'] - 1


parameters = {'C':[1e-4,1e-3,1e-2,1e-1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1e0,3],'max_iter':[1,5,70,100,130,150,170, 180, 200, 300,500,1000,1200,1400,1600,1800,2000,2200,2400,2600,3000],'l1_ratio':[1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]}
lg_clf = LogisticRegression(random_state=1, solver='saga',n_jobs=-1, penalty='elasticnet' )
grid_clf = GridSearchCV(lg_clf, parameters, scoring='roc_auc', n_jobs=-1, cv=10)
grid_clf.fit(X,Y)

f= open("data/PD_Nalls_90_58Conly_full_beta_lg_C0.6elast0.1max130_04122020_results.txt","w+")
# PD_Nalls_90_58Conly_full_beta_lg_C0.6elast0.1max130_04122020_results.txt contains the predictor performance information

f.write('PD_Nalls_90_58Conly_full_beta_lg_C0.6elast0.1max130_04122020\n')
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
model_weights.to_csv('data/PD_Nalls_90_58Conly_full_beta_lg_C0.6elast0.1max130_04122020weights.txt', sep='\t',index=False,line_terminator='\n')

import pickle
filename = 'data/PD_Nalls_90_58Conly_full_beta_lg_C0.6elast0.1max130_04122020.sav'
pickle.dump(lg_clf_best_grid, open(filename, 'wb'))

model_com = model_weights.loc[model_weights['Weight'] != 0 ]

model_com['Data_feature'].to_csv('data/PD_Nalls_90_58Conly_full_beta_lg_C0.6elast0.1max130_04122020features.txt', sep='\t',header=False,index=False,line_terminator='\n')

























parameters = {'C':[1],'max_iter':[ 500],'l1_ratio':[0.2]}
#parameters = {'C':[0.1,0.3,0.5,0.8,1,5],'max_iter':[ 200,500,600,700,800],'l1_ratio':[1,0.9,0.8,0.7,0.6,0.4,0.2,0.1,0]}
#parameters = {'C':[1,10,15,20,25,30,100],'max_iter':[ 200,500,1000,1200,1400],'l1_ratio':[1,0.9,0.8,0.7,0.6,0.4,0.2,0]}
#parameters = {'C':[1e-4,1e-3,1e-2,1e-1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1e0,3,5,7,8,10,15,20,25,30,40,50,100],'max_iter':[1,5,70,100,130,150,170, 180, 200, 300,500,1000,1200,1400,1600,1800,2000,2200,2400,2600,3000],'l1_ratio':[1,0.9,0.8,0.7,0.6,]}
lg_clf = LogisticRegression(random_state=1, solver='saga',n_jobs=-1, penalty='elasticnet' )
grid_clf = GridSearchCV(lg_clf, parameters, scoring='roc_auc', n_jobs=-1, cv=10)
grid_clf.fit(X,Y)

X.shape
grid_clf.best_score_
grid_clf.best_estimator_
grid_clf.best_params_

lg_clf_best_grid = grid_clf.best_estimator_

roc_auc_score(Y, lg_clf_best_grid.predict_proba(X)[:,1])

np.sum(lg_clf_best_grid.coef_[0,:] != 0)

GridSearchCV(cv=10,
             estimator=LogisticRegression(n_jobs=-1, penalty='elasticnet',
                                          random_state=1, solver='saga'),
             n_jobs=-1,
             param_grid={'C': [1], 'l1_ratio': [0.2], 'max_iter': [500]},
             scoring='roc_auc')
>>> X.shape
(4366, 920)
>>> grid_clf.best_score_
0.5960417478186288
>>> grid_clf.best_estimator_
LogisticRegression(C=1, l1_ratio=0.2, max_iter=500, n_jobs=-1,
                   penalty='elasticnet', random_state=1, solver='saga')
>>> grid_clf.best_params_
{'C': 1, 'l1_ratio': 0.2, 'max_iter': 500}
>>>
>>> lg_clf_best_grid = grid_clf.best_estimator_
>>>
>>> roc_auc_score(Y, lg_clf_best_grid.predict_proba(X)[:,1])
0.6041387433491735
>>>
>>> np.sum(lg_clf_best_grid.coef_[0,:] != 0)
153


filename = 'WTCCC_58Conly_88SNPs_full_tissue_specific_beta_table_lgC1elast0.2max500.sav'
pickle.dump(lg_clf_best_grid, open(filename, 'wb'))

--------------------------------------------------------------------------------------------------------------------------
parameters = {'C':[0.1,0.3,0.5,0.8,1,5],'max_iter':[ 200,500,600,700,800],'l1_ratio':[1,0.9,0.8,0.7,0.6,0.4,0.2,0.1,0]}
lg_clf = LogisticRegression(random_state=1, solver='saga',n_jobs=-1, penalty='elasticnet' )
grid_clf = GridSearchCV(lg_clf, parameters, scoring='roc_auc', n_jobs=-1, cv=10)
grid_clf.fit(X,Y)

X.shape
grid_clf.best_score_
grid_clf.best_estimator_
grid_clf.best_params_

lg_clf_best_grid = grid_clf.best_estimator_

roc_auc_score(Y, lg_clf_best_grid.predict_proba(X)[:,1])

np.sum(lg_clf_best_grid.coef_[0,:] != 0)

GridSearchCV(cv=10,
             estimator=LogisticRegression(n_jobs=-1, penalty='elasticnet',
                                          random_state=1, solver='saga'),
             n_jobs=-1,
             param_grid={'C': [0.1, 0.3, 0.5, 0.8, 1, 5],
                         'l1_ratio': [1, 0.9, 0.8, 0.7, 0.6, 0.4, 0.2, 0.1, 0],
                         'max_iter': [200, 500, 600, 700, 800]},
             scoring='roc_auc')
>>> X.shape
(4366, 920)
>>> grid_clf.best_score_
0.596348168708415
>>> grid_clf.best_estimator_
LogisticRegression(C=0.3, l1_ratio=0, max_iter=200, n_jobs=-1,
                   penalty='elasticnet', random_state=1, solver='saga')
>>> grid_clf.best_params_
{'C': 0.3, 'l1_ratio': 0, 'max_iter': 200}
>>>
>>> lg_clf_best_grid = grid_clf.best_estimator_
>>>
>>> roc_auc_score(Y, lg_clf_best_grid.predict_proba(X)[:,1])
0.604111040769368
>>>
>>> np.sum(lg_clf_best_grid.coef_[0,:] != 0)
920

-------------------------------------------------------------------------------------------------------------

parameters = {'C':[0.1,0.3,0.5,1,5],'max_iter':[ 50,100,200,300],'l1_ratio':[1,0.9,0.8,0.7,0.6,0.4,0.2,0.1]}
lg_clf = LogisticRegression(random_state=1, solver='saga',n_jobs=-1, penalty='elasticnet' )
grid_clf = GridSearchCV(lg_clf, parameters, scoring='roc_auc', n_jobs=-1, cv=10)
grid_clf.fit(X,Y)

GridSearchCV(cv=10,
             estimator=LogisticRegression(n_jobs=-1, penalty='elasticnet',
                                          random_state=1, solver='saga'),
             n_jobs=-1,
             param_grid={'C': [0.1, 0.3, 0.5, 1, 5],
                         'l1_ratio': [1, 0.9, 0.8, 0.7, 0.6, 0.4, 0.2, 0.1],
                         'max_iter': [50, 100, 200, 300]},
             scoring='roc_auc')
>>> X.shape
(4366, 920)
>>> grid_clf.best_score_
0.5961828064570824
>>> grid_clf.best_estimator_
LogisticRegression(C=5, l1_ratio=0.1, n_jobs=-1, penalty='elasticnet',
                   random_state=1, solver='saga')
>>> grid_clf.best_params_
{'C': 5, 'l1_ratio': 0.1, 'max_iter': 100}
>>>
>>> lg_clf_best_grid = grid_clf.best_estimator_
>>>
>>> roc_auc_score(Y, lg_clf_best_grid.predict_proba(X)[:,1])
0.6040410669223691
>>>
>>> np.sum(lg_clf_best_grid.coef_[0,:] != 0)
902
>>>

parameters = {'C':[0.3],'max_iter':[ 50,100,200,300],'l1_ratio':[1]}
lg_clf = LogisticRegression(random_state=1, solver='saga',n_jobs=-1, penalty='elasticnet' )
grid_clf = GridSearchCV(lg_clf, parameters, scoring='roc_auc', n_jobs=-1, cv=10)
grid_clf.fit(X,Y)

GridSearchCV(cv=10,
             estimator=LogisticRegression(n_jobs=-1, penalty='elasticnet',
                                          random_state=1, solver='saga'),
             n_jobs=-1,
             param_grid={'C': [0.3], 'l1_ratio': [1],
                         'max_iter': [50, 100, 200, 300]},
             scoring='roc_auc')
>>> X.shape
(4366, 920)
>>> grid_clf.best_score_
0.595469549578393
>>> grid_clf.best_estimator_
LogisticRegression(C=0.3, l1_ratio=1, max_iter=300, n_jobs=-1,
                   penalty='elasticnet', random_state=1, solver='saga')
>>> grid_clf.best_params_
{'C': 0.3, 'l1_ratio': 1, 'max_iter': 300}
>>>
>>> lg_clf_best_grid = grid_clf.best_estimator_
>>>
>>> roc_auc_score(Y, lg_clf_best_grid.predict_proba(X)[:,1])
0.6037223217013401
>>>
>>> np.sum(lg_clf_best_grid.coef_[0,:] != 0)
31

parameters = {'C':[0.3],'max_iter':[ 50,100,200,300],'l1_ratio':[0.5]}
lg_clf = LogisticRegression(random_state=1, solver='saga',n_jobs=-1, penalty='elasticnet' )
grid_clf = GridSearchCV(lg_clf, parameters, scoring='roc_auc', n_jobs=-1, cv=10)
grid_clf.fit(X,Y)

GridSearchCV(cv=10,
             estimator=LogisticRegression(n_jobs=-1, penalty='elasticnet',
                                          random_state=1, solver='saga'),
             n_jobs=-1,
             param_grid={'C': [0.3], 'l1_ratio': [0.5],
                         'max_iter': [50, 100, 200, 300]},
             scoring='roc_auc')
>>> X.shape
(4366, 920)
>>> grid_clf.best_score_
0.5956684538893118
>>> grid_clf.best_estimator_
LogisticRegression(C=0.3, l1_ratio=0.5, max_iter=300, n_jobs=-1,
                   penalty='elasticnet', random_state=1, solver='saga')
>>> grid_clf.best_params_
{'C': 0.3, 'l1_ratio': 0.5, 'max_iter': 300}
>>>
>>> lg_clf_best_grid = grid_clf.best_estimator_
>>>
>>> roc_auc_score(Y, lg_clf_best_grid.predict_proba(X)[:,1])
0.603953434060355
>>>
>>> np.sum(lg_clf_best_grid.coef_[0,:] != 0)
51
>>>

parameters = {'C':[0.3],'max_iter':[ 50,100,200,300,400],'l1_ratio':[0.2]}
lg_clf = LogisticRegression(random_state=1, solver='saga',n_jobs=-1, penalty='elasticnet' )
grid_clf = GridSearchCV(lg_clf, parameters, scoring='roc_auc', n_jobs=-1, cv=10)
grid_clf.fit(X,Y)
GridSearchCV(cv=10,
             estimator=LogisticRegression(n_jobs=-1, penalty='elasticnet',
                                          random_state=1, solver='saga'),
             n_jobs=-1,
             param_grid={'C': [0.3], 'l1_ratio': [0.2],
                         'max_iter': [50, 100, 200, 300, 400]},
             scoring='roc_auc')
>>> X.shape
(4366, 920)
>>> grid_clf.best_score_
0.596065924842533
>>> grid_clf.best_estimator_
LogisticRegression(C=0.3, l1_ratio=0.2, max_iter=400, n_jobs=-1,
                   penalty='elasticnet', random_state=1, solver='saga')
>>> grid_clf.best_params_
{'C': 0.3, 'l1_ratio': 0.2, 'max_iter': 400}
>>>
>>> lg_clf_best_grid = grid_clf.best_estimator_
>>>
>>> roc_auc_score(Y, lg_clf_best_grid.predict_proba(X)[:,1])
0.6040672243383609
>>>
>>> np.sum(lg_clf_best_grid.coef_[0,:] != 0)
92


parameters = {'C':[0.3],'max_iter':[ 50,100,200,300,400,500],'l1_ratio':[0.1]}
lg_clf = LogisticRegression(random_state=1, solver='saga',n_jobs=-1, penalty='elasticnet' )
grid_clf = GridSearchCV(lg_clf, parameters, scoring='roc_auc', n_jobs=-1, cv=10)
grid_clf.fit(X,Y)

GridSearchCV(cv=10,
             estimator=LogisticRegression(n_jobs=-1, penalty='elasticnet',
                                          random_state=1, solver='saga'),
             n_jobs=-1,
             param_grid={'C': [0.3], 'l1_ratio': [0.1],
                         'max_iter': [50, 100, 200, 300, 400, 500]},
             scoring='roc_auc')
>>> X.shape
(4366, 920)
>>> grid_clf.best_score_
0.5961629913373641
>>> grid_clf.best_estimator_
LogisticRegression(C=0.3, l1_ratio=0.1, max_iter=400, n_jobs=-1,
                   penalty='elasticnet', random_state=1, solver='saga')
>>> grid_clf.best_params_
{'C': 0.3, 'l1_ratio': 0.1, 'max_iter': 400}
>>>
>>> lg_clf_best_grid = grid_clf.best_estimator_
>>>
>>> roc_auc_score(Y, lg_clf_best_grid.predict_proba(X)[:,1])
0.6040673347072048
>>>
>>> np.sum(lg_clf_best_grid.coef_[0,:] != 0)
150
