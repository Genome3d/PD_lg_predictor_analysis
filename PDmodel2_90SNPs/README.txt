Parkinson's disease (PD) PD model-2 regularized logistic regression predictor model from the 90 PD SNPs of Nalls et al.

Python 3.7.3 tsfresh 0.16.0 scikit-learn 0.23.2

unzip all the zip files under data/ before running any python scripts. 
Some required data may not be available. Please double check before running the python scripts.

This folder contains 4 python scripts for creating and validating the PD model-2 regularized logistic regression predictor developed from our study. 
The scripts should be run by python directly as 4 sequential steps. All the required files are location in data/ directory. 
The python_scripts_of_PD_manuscript/ diectory contains all the text files of commands used for developing the predictor models. They are only used for references. 

For using the default files at data/, your should start with B_PD_90Nalls_create_PD_model-2.py 
and skip A_PD_90Nalls_create_eQTL_table.py. 

You can start with running A_PD_90Nalls_create_eQTL_table.py (optional):

For running A_PD_90Nalls_create_eQTL_table.py to create a individual tissue specific eQTL effect table, 
it is required to have your own individual genetype data contain T1D GWAS SNPs in plink bed format with three files (bed,fam,bin) to be converted to a .raw text file. 
The pink command is as the following:

plink -bfile your_genotype_bed_file --recode A --out individual_genotype_table

After running this command, individual_genotype_table.raw is created and be placed in data/. 
Due to the data sharing agreements, we could not provide any indiviual genotype data from Wellcome Trust Case and Control Consortium, UK Biobank and dbGap NeuroX-dbGap. 
Nevertheless, you can go to https://www.wtccc.org.uk/ and https://www.ukbiobank.ac.uk/ 
and https://www.ncbi.nlm.nih.gov/projects/gap/cgi-bin/study.cgi?study_id=phs000918.v1.p1for applying their genotype data.
