import os
from rdkit import Chem
import os, pickle
pdb_smiles={}

# 文件夹名称数组（假设是相对路径，若为绝对路径直接写全路径即可）
#folder_names = ['3bho']
folder_names = ['3bho', '1h07', '2vr0', '2fm5', '6h7k', '3v9b', '4dgo', '1mue', '6fim', '2q2n', '5v8h', '5v8j', '2fou', '4otw', '4knz', '3lp2', '1sl3', '2foy', '3lp1', '1ksn', '1a7x', '5wjj', '5lwe', '1nu1', '3kck', '2pll', '2fov', '6gu6', '5ohj', '4kcx']
#D:\browser download\PDBbind_v2020_refined\refined-set
#D:\browser download\PDBbind_v2020_other_PL\PDBbind_v2020_other_PL\v2020-other-PL
dir1="D:\\browserdownload\\PDBbind_v2020_refined\\refined-set"
dir2="D:\\browserdownload\\PDBbind_v2020_other_PL\\PDBbind_v2020_other_PL\\v2020-other-PL"
x=0
y=0
err=[]
for i in folder_names:
    dirs1 = os.path.join(dir1,i)
    dirs2 = os.path.join(dir2,i)
    try:
        if(os.path.exists(dirs1)):
            for _, _, dirnames in os.walk(dirs1):
                for dirname in dirnames:
                        if 'ligand.mol2' in dirname:
                            filedir = os.path.join(dirs1,dirname)
                            print("**************************",filedir)
                            mol = Chem.MolFromMol2File(filedir)
                            smile = Chem.MolToSmiles(mol)
                            pdb_smiles[i] = smile
                            x=x+1
        if(os.path.exists(dirs2)):
            for _, _, dirnames in os.walk(dirs2):
                for dirname in dirnames:
                        if 'ligand.mol2' in dirname:
                            filedir = os.path.join(dirs2,dirname)
                            print("**************************",filedir)
                            mol = Chem.MolFromMol2File(filedir)
                            smile = Chem.MolToSmiles(mol)
                            pdb_smiles[i] = smile
                            y=y+1
    except Exception as e: 
        err.append(dirname)
        
print(pdb_smiles)

                    
print(x)    
print(y) 
print(err)