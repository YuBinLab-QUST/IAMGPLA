
from rdkit import Chem
import pickle
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles


f = open(r'C://Users//25319//Desktop//renew//pdb_smiles.pkl', "rb")#文件所在路径
pdb_smiles = pickle.load(f, encoding='latin1')
print(len(pdb_smiles))


f = open(r'C://Users//25319//Desktop//renew//pdbid_smile2020.pkl', "rb")#文件所在路径
pdb_smiles = pickle.load(f, encoding='latin1')
print(len(pdb_smiles))

f = open(r'C://Users//25319//Desktop//renew//Scaffold_refined2020_data.pkl', "rb")#文件所在路径
refined_set_ids = pickle.load(f, encoding='latin1')
print(len(refined_set_ids))

