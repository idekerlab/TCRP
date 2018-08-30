import os
import pandas as pd
import numpy as np
import sys
import pickle
from scipy.spatial.distance import cdist
import math
import itertools

cell_line_detail_file = data_file + 'Cell_Lines_Details.csv'
cell_line_legend = pd.read_csv(cell_line_detail_file, sep='\t', index_col=1)
cell_line_list = list( cell_line_legend.index )

tissue_map = {}

for cell_line in cell_line_list:
	
	tissue = cell_line_legend.loc[cell_line,'Site']
	
	if tissue not in tissue_map:
		tissue_map[tissue] = []
		tissue_map[tissue].append(cell_line)
		
drug_cell_line_file = data_file + 'v17_fitted_dose_response.csv'

drugs = pd.read_csv(drug_cell_line_file, sep='\t',index_col=2)

drugs_legend = pd.read_csv('sanger_cell_line_data/Screened_Compounds.csv', sep='\t', index_col=0)

drug2id_mapping = {}

for index in list(drugs_legend.index):
	drug_name = drugs_legend.loc[index,'DRUG NAME']
	drug2id_mapping[ drug_name ] = index

drug = sys.argv[1]
gene = sys.argv[2]
tissue = sys.argv[3]

if drug not in drug2id_mapping:
	print 'drug name wrong', drug
	sys.exit(1)
		
cell_line_drug_matrix = drugs.loc[ drugs['DRUG_ID'] == drug2id_mapping[drug] ]
drug_tissue_map = {}
	
for tissue, tissue_cell_line_list in tissue_map.items():

tissue_cell_line_list = tissue_map[ tissue ]
drug_specific_cell_line = set( cell_line_drug_matrix.index ) & set( tissue_cell_line_list )
		
		#label = cell_line_drug_matrix.loc[ drug_specific_cell_line,'LN_IC50'].values

		drug_tissue_map[tissue] = drug_specific_cell_line
	
		drug	
import multiprocessing
p = multiprocessing.Pool(30)

param_list = []

for gene in crispr_gene_list:
	param_list.append((gene))

	for tissue, tissue_cell_line_list in tissue_map.items():

		
