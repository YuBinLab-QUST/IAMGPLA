main(input_path,out_path)：  #主函数
input_path 数据集文件夹
out_path 预处理输出文件夹

feature_cal(f_g,input_path, out_path, mol_dict, mapping_dict)   #特征计算
f_g = 蛋白质pdb文件文件夹
mapping_dict   蛋白质id与配体id对应文件
mol_dict  配体   配体id和rdkit.mol对应文件

def load_blosum62(input_path):  blosum62对应文件
def atom_features(atom):  原子特征
def bond_features(bond):  键特征
def Mol2Graph(mol):  构图函数
