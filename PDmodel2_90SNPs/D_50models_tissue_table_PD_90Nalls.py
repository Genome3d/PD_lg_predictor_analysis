# This program to combine the 50 model results from C_10fold_PDmodel-2_hyperparameters_50_predictors 
# and create a tissue model weights matrix of the 50 predictor built with PD_model-2 hyperparameters

# Input files : 10fold_PDModel-2_hyperparameteres_model{count}weights.txt where count = 1 to 50 contains the model components and their weights for each randomized model

# Output file : PD_model-2_tissue_results_50models.txt a tissue model weights matrix of the 50 predictor created with PD_model-1 hyperparameters

import numpy as np
import pandas as pd

items = []
table = {}
interlines = []
intralines = []
col = []
PriGene = { }
Geneid = []

MaRfrid = { }
MaRCor = { }

def load_table(datafile):
	
	TSG_weights = {}
	TSG_count = {}
	TSG_Var = {}
	SNP_weights = float(0)
	total_TSG = 0
	SNP_count = 0
	SNP_Var = float(0)
	abs_value = float(0)
	sitems = [ ]
	titems = [ ]
	fitems = [ ]
	f2items = [ ]
	PriGene = { }
	SNPid = {}
	MaRfrid = { }
	MaRCor = { }
	interlines = []
	count = 0
	
	
	inputfile = open(datafile, 'r')
	while True:
		input_line = inputfile.readline()
		if count == 0:
			count += 1
			input_line = inputfile.readline()
			TSG_weights['file']= datafile
		input_line = input_line[:-1]
		if not input_line:
			break
		else:
			items = input_line.split('\t')
			
			TSG_SNPs = items[0].strip().split('--')
			tissue_SNP = TSG_SNPs[0].strip()
			if (tissue_SNP[:2] == 'rs') and ('SNPs' not in TSG_weights):
				TSG_weights['SNPs'] = float(0)
				TSG_count['SNPs'] = 0
				TSG_Var['SNPs'] = float(0)
			if (tissue_SNP not in TSG_weights) and (tissue_SNP[:2] != 'rs'):
				TSG_weights[tissue_SNP] = float(0)
				TSG_count[tissue_SNP] = 0
				TSG_Var[tissue_SNP] = float(0)
			
			
			if tissue_SNP[:2] == 'rs' :
				TSG_weights['SNPs'] += abs(float(items[1].strip()))
				if abs(float(items[1].strip())) != float(0): TSG_count['SNPs'] += 1
				TSG_Var['SNPs'] += (float(items[1].strip()))**2
			else:
				TSG_weights[tissue_SNP] += abs(float(items[1].strip()))
				if abs(float(items[1].strip())) != float(0): TSG_count[tissue_SNP] += 1
				TSG_Var[tissue_SNP] += (float(items[1].strip()))**2
				
				
			
		



	inputfile.close()
	return TSG_weights


def save_items(filename , organ_value):
	newfile = filename + "_50TissEffects.txt"
	outputfile = open(newfile, 'w')
	outputfile.write(filename)
	outputfile.write('\n')
	for x in organ_value:
		outputfile.write(str(x))
		outputfile.write('\n')
	outputfile.close()
	

interlines = []

abs_value = float(0)
organ_value = {}


for Num in range(50):
	Num1 = Num + 1
	filename = 'data/10fold_PDModel-2_hyperparameteres_model' + str(Num1) + 'weights.txt'
	organ_value = load_table(filename)
	interlines.append(organ_value)

organ_data = pd.DataFrame(interlines)

organ_data.to_csv('data/PD_model-2_tissue_results_50models.txt',  index = None, sep='\t', header=True) 

print ("Write file !!")






