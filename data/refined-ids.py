
import os,pickle
import pandas as pd

dir="D:\\browserdownload\\PDBbind_v2020_refined\\refined-set"
refined_set_ids=[]
for _, dirnames, _ in os.walk(dir):
    for i in dirnames:
        refined_set_ids.append(i)
print(len(refined_set_ids))



f = open(r'C://Users//25319//Desktop//renew//pdbid_smile2020.pkl', "rb")#文件所在路径
pdb_smiles = pickle.load(f, encoding='latin1')
refined_set_id=[]
for i in pdb_smiles.keys():
    if i in refined_set_ids:
        refined_set_id.append(i)

print(refined_set_id)

""" with open("C://Users//25319//Desktop//renew//refined_set_ids.pkl", "wb") as f:
    pickle.dump(refined_set_id, f) 
print("ok")  """

""" 5318
4741
ok """