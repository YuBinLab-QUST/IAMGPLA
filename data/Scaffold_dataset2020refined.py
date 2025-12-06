
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
    print(scaffold_sets)


    f = open(r'C://Users//25319//Desktop//renew//refined_set_ids.pkl', "rb")#文件所在路径
    refined_set_ids = pickle.load(f, encoding='latin1')


    refined_set_ind=[]

    for ind, id in enumerate(dataset.keys()):
        if id in refined_set_ids:
            refined_set_ind.append(ind)
         


    print("refined_set_ind:",len(refined_set_ind))


  
    refined_set_ind = set(refined_set_ind)  # 转为集合提升效率

    # 步骤1：计算每个子列表与目标数组的交集，并过滤空列表
    scaffold_sets = [
        [x for x in sublist if x in refined_set_ind]  # 计算单个子列表的交集
        for sublist in scaffold_sets
        if any(x in refined_set_ind for x in sublist)  # 只保留非空交集的子列表
    ]

    # 步骤2：计算所有交集元素的总个数
    total_count = sum(len(sublist) for sublist in scaffold_sets)
    print(total_count)



    train_cutoff = train_size * total_count
    valid_cutoff = (train_size + valid_size) * total_count
    train_inds = []
    valid_inds = []
    test_inds = []

  
    for scaffold_set in scaffold_sets:
        if len(train_inds) + len(scaffold_set) > train_cutoff:
            if len(train_inds) + len(valid_inds) + len(scaffold_set) > valid_cutoff:
                test_inds += scaffold_set
            else:
                valid_inds += scaffold_set
        else:
            train_inds += scaffold_set
    return train_inds, valid_inds, test_inds


f = open(r'C://Users//25319//Desktop//renew//pdbid_smile2020.pkl', "rb")#文件所在路径
pdb_smiles = pickle.load(f, encoding='latin1')
print(len(pdb_smiles))
train_inds, valid_inds, test_inds=scaffold_split(pdb_smiles, 0.1,0.2)
data=[]
data.append(train_inds)
data.append(valid_inds)
data.append(test_inds)
print(len(train_inds))
print(len(valid_inds))
print(len(test_inds))

with open("C://Users//25319//Desktop//renew//Scaffold_refined2020_data.pkl", "wb") as f:
    pickle.dump(data, f) 
print("ok") 


"""
# 原始数据
original_list = [[1,2,3], [5,7], [2,4]]  # 新增[2,4]用于测试空交集场景
target_arr = [1,5,7]
target_set = set(target_arr)  # 转为集合提升效率

# 步骤1：计算每个子列表与目标数组的交集，并过滤空列表
non_empty_intersections = [
    [x for x in sublist if x in target_set]  # 计算单个子列表的交集
    for sublist in original_list
    if any(x in target_set for x in sublist)  # 只保留非空交集的子列表
]

# 步骤2：计算所有交集元素的总个数
total_count = sum(len(sublist) for sublist in non_empty_intersections)

# 输出结果
print(non_empty_intersections)
print(total_count)
 """

""" 4737
3315
474
948
ok """