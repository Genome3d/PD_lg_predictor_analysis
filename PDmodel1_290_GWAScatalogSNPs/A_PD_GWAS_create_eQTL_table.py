# create a genotype table
#
# Assume your_genotype_bed_file is your plink genotype data contain thePD GWAS 290 SNPs in my study only
# rs504594 was always added into our analysis for one (rs112485576) of the 290 PD SNPs was meraged to rs504594
# your_genotype_bed_file is in plink bed format with three files (bed,fam,bin)
# They need to be converted to a text table format (.raw)

# plink -bfile your_genotype_bed_file --recode A --out individual_genotype_table



#--------------------------------------------------------------------------------------
# python program for creating the PD tissue specific eQTL table "Weighted_eQTL_matrix.txt" from your own genotype data
#
# Inputs: Tissue_SNP_Gene_mapping.txt, individual_genotype_table.raw
# Outputs: Weighted_eQTL_matrix.txt




# python program for creating the Tissue specific eQTL expression table.

import numpy as np
import pandas as pd

TSG_list = pd.read_csv('data/Tissue_SNP_Gene_mapping.txt',sep='\t')
eQTL_table = pd.read_csv('data/individual_genotype_table.raw',sep = " ")

tmp_eQTL_table = eQTL_table.copy()
samples_header = list (eQTL_table.columns[0:6])
SNP_w_list = list(eQTL_table.columns[6:])
tmp_SNP_w_list = SNP_w_list.copy()
TSG_cols = []
drop_SNPs = []

for rsSNP_w in tmp_SNP_w_list :
	rsSNP = rsSNP_w.split('_')[0]
	tmp_TSGs = list ( TSG_list.Tissue[TSG_list.SNP == rsSNP] + "--" + rsSNP_w + "--" + TSG_list.Gene_Name[TSG_list.SNP == rsSNP])
	tmp_Effects = list ( TSG_list.Effect_Size[TSG_list.SNP == rsSNP])
	if len(tmp_Effects) != 0 :
		count = 0
		for TSG in tmp_TSGs:
			tmp_eQTL_table[TSG] = tmp_eQTL_table[rsSNP_w] * tmp_Effects[count]
			count = count + 1
			TSG_cols.append(TSG)
		drop_SNPs.append(rsSNP_w)

for rsSNP_w in drop_SNPs:
	SNP_w_list.remove(rsSNP_w)
	
TSG_col_sorted = sorted(TSG_cols)
new_eQTL_table = tmp_eQTL_table[samples_header + TSG_col_sorted + SNP_w_list]
new_eQTL_table.to_csv('data/Weighted_eQTL_matrix.txt', sep='\t', index=False)
# Weighted_eQTL_matrix.txt is the individual tissue specific eQTL effect table (referring to the Supplementary Table of the PD manuscript)
# The matrix contains the information integrated from the idividual GWAS T1D SNPs and their tissue specific effects.
