import pandas as pd
import pickle
f = open(r'C://Users//25319//Desktop//pdb_smiles.pkl', "rb")#文件所在路径   pdbbind所有id及其对应smile  ‪C:\Users\25319\Desktop\renew\pdb_smiles.pkl
pdb_smiles = pickle.load(f, encoding='latin1')
print(len(pdb_smiles))



pdbids=[]
def ids(pdbid):
    for i, row in pdbid.iterrows():
        cid, pKa = row['id'], float(row['affinity'])
        pdbids.append(cid)
    return pdbids
pdbid1 = pd.read_csv('C://Users//25319//Desktop//aff_20_8.csv')
""" pdbid2 = pd.read_csv('C://Users//25319//Desktop//aff_16_8.csv')
pdbid3 = pd.read_csv('C://Users//25319//Desktop//aff_csar_8.csv') """
pdbids=ids(pdbid1)
print(len(pdbids))

pdbid_smile = {}

for i in pdbids:
    if i in pdb_smiles.keys():
        pdbid_smile[i]=pdb_smiles[i]
    else: pdbid_smile[i]="err"

with open("C://Users//25319//Desktop//pdbid_smile.pkl", "wb") as f:# 训练集id及其对应smile  pdbid_smile2020.pkl
    pickle.dump(pdbid_smile, f) 
print("ok") 

print(len(pdbid_smile) )


""" 
19398
17381
ok
17351
30
['3bho', '1h07', '2vr0', '2fm5', '6h7k', '3v9b', '4dgo', '1mue', '6fim', '2q2n', '5v8h', '5v8j', '2fou', '4otw', '4knz', '3lp2', '1sl3', '2foy', '3lp1', '1ksn', '1a7x', '5wjj', '5lwe', '1nu1', '3kck', 
'2pll', '2fov', '6gu6', '5ohj', '4kcx'] """

