# Combine Weighted_eQTL_matrix_1.txt and Weighted_eQTL_matrix_2.txt into Weighted_eQTL_matrix.txt
import numpy as np
import pandas as pd

temp_tabletop = pd.read_csv("Weighted_eQTL_matrix_1.txt", sep="\t")
temp_tablelow = pd.read_csv("Weighted_eQTL_matrix_2.txt", sep="\t")
full_data = pd.concat([temp_tabletop,temp_tablelow],ignore_index=True)
full_data.to_csv("Weighted_eQTL_matrix.txt", sep='\t', index=False)