import pandas as pd
import pickle
f = open(r'C://Users//25319//Desktop//data_8.pkl', "rb")#文件所在路径
pdb_smiles = pickle.load(f, encoding='latin1')
print(len(pdb_smiles))
print(len(pdb_smiles[0]))
print(len(pdb_smiles[1]))
print(len(pdb_smiles[2]))
