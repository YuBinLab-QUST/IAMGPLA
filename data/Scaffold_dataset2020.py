
from rdkit import Chem
import pickle
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

def _generate_scaffold(smiles, include_chirality=False):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold


def generate_scaffolds(dataset):
    scaffolds = {}
    err=[]
    print(len(dataset))
    pdbsmiles=dataset.values()

    for ind, smiles in enumerate(pdbsmiles):
        
        try:
            scaffold = _generate_scaffold(smiles)
            if scaffold not in scaffolds:
                scaffolds[scaffold] = [ind]
            else:
                scaffolds[scaffold].append(ind)
        except Exception as e: 
            err.append(ind)
    print(err)
    print(len(scaffolds))

    # Sort from largest to smallest scaffold sets
    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]
    return scaffold_sets


def scaffold_split(dataset, valid_size, test_size):
    train_size = 1.0 - valid_size - test_size
    scaffold_sets = generate_scaffolds(dataset)

    train_cutoff = train_size * len(dataset)
    valid_cutoff = (train_size + valid_size) * len(dataset)
    train_inds = []
    valid_inds = []
    test_inds = []

    print("About to sort in scaffold sets")
    for scaffold_set in scaffold_sets:
        if len(train_inds) + len(scaffold_set) > train_cutoff:
            if len(train_inds) + len(valid_inds) + len(scaffold_set) > valid_cutoff:
                test_inds += scaffold_set
            else:
                valid_inds += scaffold_set
        else:
            train_inds += scaffold_set
    return train_inds, valid_inds, test_inds


f = open(r'C://Users//25319//Desktop//renew//pdbid_smile2020.pkl', "rb")#文件所在路径    训练集
pdb_smiles = pickle.load(f, encoding='latin1')
print(len(pdb_smiles))
train_inds, valid_inds, test_inds=scaffold_split(pdb_smiles, 0.1,0.2)
data=[]
data.append(train_inds)
data.append(valid_inds)
data.append(test_inds)

with open("C://Users//25319//Desktop//renew//Scaffold_data2020.pkl", "wb") as f:
    pickle.dump(data, f) 
print("ok") 


