
import numpy as np
import pandas as pd
import torch

# from tqdm import tqdm_notebook as tqdm
import pickle

import dgl

from tqdm.auto import tqdm


import glob
import networkx as nx
import os
import re
import math
import cmath
from einops import rearrange
from collections import defaultdict
from rdkit import Chem

import torch


namess=[]

elem_list = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm', 'Os', 'Ir', 'Ce','Gd','Ga','Cs', 'unknown']
aa_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
atom_fdim = len(elem_list) + 6 + 6 + 6 + 1   ### total lengh: 82   len(elem_list):63
bond_fdim = 6
max_nb = 6

input_path = "/opt/data/private/process/pdbbind2020"
out_path = "/opt/data/private/process/pdbbind_porcess6/output"
f_g = glob.glob('/opt/data/private/process/pdbbind2020/*_protein.pdb')
# print(f_g[:3]+f_g[-3:])

mapping_dict={}
with open(r'/opt/data/private/process/pdbbind_porcess6/input/mea', "r") as g:
    for line in g:
        mapping_dict[line.strip().split(' ')[0]]=line.strip().split(' ')[1]
with open('/opt/data/private/process/pdbbind_porcess6/input/mol','rb') as f:
        mol_dict = pickle.load(f)
atom_dict = defaultdict(lambda: len(atom_dict))
bond_dict = defaultdict(lambda: len(bond_dict))

def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom):
    return np.array(onek_encoding_unk(atom.GetSymbol(), elem_list) 
            + onek_encoding_unk(atom.GetDegree(), [0,1,2,3,4,5]) 
            + onek_encoding_unk(atom.GetExplicitValence(), [1,2,3,4,5,6])
            + onek_encoding_unk(atom.GetImplicitValence(), [0,1,2,3,4,5])
            + [atom.GetIsAromatic()], dtype=np.float32)


def bond_features(bond):
    bt = bond.GetBondType()
    return np.array([bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, \
    bt == Chem.rdchem.BondType.AROMATIC, bond.GetIsConjugated(), bond.IsInRing()], dtype=np.float32)


def Mol2Graph(mol):    ### !!!! the special atoms and bonds in mol files are different from other condition ???? 
    # convert molecule to GNN input
    idxfunc=lambda x:x.GetIdx()

    n_atoms = mol.GetNumAtoms()
    assert mol.GetNumBonds() >= 0

    n_bonds = max(mol.GetNumBonds(), 1)
    fatoms = np.zeros((n_atoms,), dtype=np.int32) #atom feature ID
    fbonds = np.zeros((n_bonds,), dtype=np.int32) #bond feature ID
    atom_nb = np.zeros((n_atoms, max_nb), dtype=np.int32)   ###(n_atoms , 6)
    bond_nb = np.zeros((n_atoms, max_nb), dtype=np.int32)   ###(n_atoms, 6)
    num_nbs = np.zeros((n_atoms,), dtype=np.int32)          ### (a_atoms)
    num_nbs_mat = np.zeros((n_atoms,max_nb), dtype=bool)    ### (n_atoms , 6)

    for atom in mol.GetAtoms():
        idx = idxfunc(atom)  ### get the atom id 
#         for i in atom_features(atom).astype(int).tolist():
#             print(i)
#         print(atom_features(atom).astype(int).tolist())
        fatoms[idx] = atom_dict[''.join(str(x) for x in atom_features(atom).astype(int).tolist())]   ### return the atoms feats in the list 
##### 000000000000000001000000000000000000000000000000000000000010000000001000100...： the key of atom_dict
    for bond in mol.GetBonds():   ### the bond form
        a1 = idxfunc(bond.GetBeginAtom())   ### !!!! getting the ids of bond start and end atoms !!!
        a2 = idxfunc(bond.GetEndAtom())
        idx = bond.GetIdx()      ### get the bond id 
        fbonds[idx] = bond_dict[''.join(str(x) for x in bond_features(bond).astype(int).tolist())]  ### list : return the bond feats 
        try:
            atom_nb[a1,num_nbs[a1]] = a2
            atom_nb[a2,num_nbs[a2]] = a1
        except:
            return [], [], [], [], []
        bond_nb[a1,num_nbs[a1]] = idx
        bond_nb[a2,num_nbs[a2]] = idx
        num_nbs[a1] += 1
        num_nbs[a2] += 1
        
    for i in range(len(num_nbs)):
        num_nbs_mat[i,:num_nbs[i]] = 1    ### the ith atoms have how many bonds: the bonds is 1

    return fatoms, fbonds, atom_nb, bond_nb, num_nbs_mat


def Batch_Mol2Graph(mol_list):
    res = list(map(lambda x:Mol2Graph(x), mol_list))
    fatom_list, fbond_list, gatom_list, gbond_list, nb_list = zip(*res)   ### 与 zip(返回元组列表 ) 相反，可理解为解压，返回二维矩阵式
    return fatom_list, fbond_list, gatom_list, gbond_list, nb_list ### return the atoms,bonds feats list; the atoms_int,bonds_about_atoms list; the atoms having the numbers of bonds 

def load_blosum62(input_path):
    blosum_dict = {}
    f = open('/opt/data/private/process/pdbbind_porcess6/input/blosum61.txt')
    lines = f.readlines()
    f.close()
    skip =1 
    for i in lines:
#         print(i)
        if skip == 1:
            skip = 0
            continue
        parsed = i.strip('\n').split()
        blosum_dict[parsed[0]] = np.array(parsed[1:]).astype(float)
    return blosum_dict
### save features in list 
        
### defined the function for feature calculation
def feature_cal(f_g,input_path, out_path, mol_dict, mapping_dict): 
    atom_dict = defaultdict(lambda: len(atom_dict))
    bond_dict = defaultdict(lambda: len(bond_dict))
    aa_codes = {
    'ALA':'A','CYS':'C','ASP':'D','GLU':'E',
    'PHE':'F','GLY':'G','HIS':'H','LYS':'K',
    'ILE':'I','LEU':'L','MET':'M','ASN':'N',
    'PRO':'P','GLN':'Q','ARG':'R','SER':'S',
    'THR':'T','VAL':'V','TYR':'Y','TRP':'W'}
    blosum_dict = load_blosum62(input_path)  ###load the evolutionary dict
# print(blosum_dict)
    list9=[]
    list99=[]
    listcol99=[]
    for i in range(0,10):
        list_column9 = ['H', str(i), ' ']
        str_name9 = ''.join(list_column9)
        list9.append(str_name9)
    # print(list9)
    for i in range(10,100):
        list_column99 = ['H', str(i)]
        str_name99 = ''.join(list_column99)
        list99.append(str_name99)
    for i in range(0,100):
        l_column99 = ['H', '  ', '.', str(i)]
        st_name99 = ''.join(l_column99)
        listcol99.append(st_name99)

    # print(atom_dict,bond_dict)
    pro_vertex=[]
    pro_edge=[]
    int_provertex=[]
    int_edge=[]
    int_ligvertex=[]
    proid=[]
    # aff=[]
    mol_inputs=[]  
    i=0
    j=0
    for fi_g in f_g:
        try:
            seq_name = os.path.basename(fi_g)[:4]
            #proid.append(seq_name)
    #         print(seq_name)
            content =[]
            record = False
            LATOM = []
            with open(r'%s/%s_ligand.mol2'%(input_path,seq_name), "r") as g:
                for line1 in g:
                    if line1.strip().split(' ')[0]=='@<TRIPOS>ATOM':
                        record = True
                    if record:
                        content.append(line1)
                    if line1.strip().split(' ')[0]=='@<TRIPOS>BOND':
                        break
            content = content[1:-1]

            i=-1
            for cont in content:
        #        print(cont)
                if cont[47:48]!='H':
                    i+=1
                    LATOM.append( [float(cont[17:26]),float(cont[26:36]),float(cont[36:46]),cont[47:48]+str(i)] )


            PATOM = {}
            with open(r'%s/%s_protein.pdb'%(input_path,seq_name), "r") as f:
                for line in f:
                    if line.strip().split(' ')[0]=='ATOM':
                        res_num = line[22:26]
                        res_type = aa_codes[line[17:20]]
                        chain = line[21:22]
                        res_id =res_type+"_"+res_num+"_"+chain
                        ele = [float(line[30:38]),float(line[38:46]),float(line[46:54]),line[77:78]]
                        if res_id not in PATOM:
                            PATOM[res_id] = [ele]
                        else:
                            PATOM[res_id].append(ele)

            key1=[]

            for item2 in LATOM:
                for key in PATOM:
                    min_dis=8
                    for item1 in PATOM[key]:
                        if item1[3]!="H":
                            dis=math.sqrt((item1[0]-item2[0])**2.0+(item1[1]-item2[1])**2.0+(item1[2]-item2[2])**2.0)
                            if dis<min_dis:
                                min_dis=dis
                                atom=item2[3]
                                dis_pl=dis
                                key1.append( [atom,key,dis_pl] )
        #                     if len(key1)==0:
        #                         key1.append( [atom,key,dis_pl] )
        #                     else:
        #                         if atom==key1[np.array(key1).shape[0]-1][0] and key==key1[np.array(key1).shape[0]-1][1]:
            inter_list=[]
            k=0
            h=0
        # intera_list=[]
            for i in range(len(key1)):
                if i ==0:
                    k+=1
                elif i !=0 and i !=len(key1)-1:
                    if key1[i][0]==key1[i-1][0] and key1[i][1]==key1[i-1][1]:
                        h+=1
                    else:
                        inter_list.append(key1[i-1])
                else:
                    if key1[i][0]==key1[i-1][0] and key1[i][1]==key1[i-1][1]:
                        inter_list.append(key1[i])
                    else:
                        inter_list.append(key1[i-1])
                        inter_list.append(key1[i])

        # print(len(key1))
        # print(key1)
        # print(len(inter_list))
        # print(inter_list)

        #########protein ligand index order
            set_p=set()
            set_l=set()
            for x in range(len(inter_list)):
                set_p.add(inter_list[x][1])
                set_l.add(inter_list[x][0])
        # for protein_res in set_p:

            list_l=list(set_l)    
            list_p=list(set_p)

            def takeSecond(elem):
                return int(elem.strip().split('_')[1].replace(" ", ""))
            list_p.sort(key=takeSecond)
            def takeNum(elem):
                return int(elem[1:])
            list_l.sort(key=takeNum)
        #     inter_list[x][1].strip().split('_')[1].replace(" ", "")
        # print(list_p)
        # print(list_l)
        #########protein ligand index order
            data=np.zeros((len(list_p),len(list_l)))
            for i in range (len(list_p)):
                for j in range (len(list_l)):
                    for m in range(len(inter_list)):
                        if list_p[i]==inter_list[m][1] and list_l[j]==inter_list[m][0]:
                            data[i][j]=inter_list[m][2]

        # print(data)
            matr =pd.DataFrame(rearrange(data,'res1 res2 -> res1 res2')) 
            matr.columns=[list_l[j][0] for j in range(len(list_l))]
            matr.index = [list_p[i] for i in range(len(list_p))] 
            matr.to_csv(f'%s/{seq_name}_int.csv'%(out_path))
        #######################################################################
            with open(r'%s/%s_protein.pdb'%(input_path,seq_name), "r") as f:
        #         print(seq_name)
        #         data = np.zeros((10,10,12))
                Natom=1
                Nresdue=1
                AA={}
                x={}
                y={}
                z={}
                for line in f:
                    if line.strip().split(' ')[0]=='ATOM':
                        if Natom ==1:
                            AA[Nresdue]=aa_codes[line[17:20]]
                            NAA = line[22:26]
                            Natom+=1
                            if 'CA' in line:
                                x[Nresdue]=line[30:38]
                                y[Nresdue]=line[38:46]
                                z[Nresdue]=line[46:54]
                        else:
                            if line[22:26]==NAA:
                                Natom+=1
                                if 'CA' in line:
                                    x[Nresdue]=line[30:38]
                                    y[Nresdue]=line[38:46]
                                    z[Nresdue]=line[46:54]
        #                         print(x[Nresdue],y[Nresdue],z[Nresdue])
                            else:
                                NAA=line[22:26]
                                Nresdue+=1
                                AA[Nresdue]=aa_codes[line[17:20]]
                                Natom+=1
                                if 'CA' in line:
                                    x[Nresdue]=line[30:38]
                                    y[Nresdue]=line[38:46]
                                    z[Nresdue]=line[46:54]
        #                         print(x[Nresdue],y[Nresdue],z[Nresdue])

                resdue_matrix = np.zeros((Nresdue,Nresdue))
                Nresdue_list=[]
                for k in range(1,Nresdue+1):
                    Nresdue_list.append(AA[k])
                for i in range(1,Nresdue+1):
                    for j in range(1,Nresdue+1):
                        #print(AA[i],AA[j])
                        dis = math.sqrt((float(x[i])-float(x[j]))**2.0+(float(y[i])-float(y[j]))**2.0+(float(z[i])-float(z[j]))**2.0)
        #                 print(dis)
                        if abs(i-j)<=1:
                            cont=0
                        if abs(i-j)>=2:     
                            if dis <=8.0:
                                cont=1
                            else:
                                cont=0
        #             res1 = {m:n for (n,m) in enumerate(Nresdue_list)}
        #             res2 = {m:n for (n,m) in enumerate(Nresdue_list)}
        #             print(res1)
                        resdue_matrix[i-1,j-1]=cont
        #         print(resdue_matrix)
                matr =pd.DataFrame(rearrange(resdue_matrix,'res1 res2 -> res1 res2')) 
                matr.columns=[AA[i] for i in range(1,Nresdue+1)]
                matr.index = [AA[j] for j in range(1,Nresdue+1)]
                matr.to_csv(f'%s/{seq_name}.contact'%(out_path))  

            with open(r'%s/%s.contact'%(out_path,seq_name), "r") as f1:
                pro_aa=[]
                Normalized_position=[]
                evolutionary_feature=[]
                for line in f1:
                    if line[0]!=',':
                        pro_aa.append(line[0])      ######get the aa list  1D
                pro_length=len(pro_aa)
                for i in range(0,pro_length):
                    position=(i+1)/pro_length
                    Normalized_position.append(position)   ######get the normalized position features     1D
                    evolutionary_feature.append(blosum_dict[pro_aa[i]])   ###get the evolutionary features   20D
                p=[]
                for i in range(0,pro_length):
                    p.append([pro_aa[i]]+[Normalized_position[i]]+[evolutionary_feature[i]])    ###get protein vertex features in contact
                #pro_vertex.append(p)   ### the pro_aa[i] save the AA types
        #         print(pro_vertex)
            f2=pd.read_csv('%s/%s.contact'%(out_path,seq_name))
            f2.drop('Unnamed: 0',axis=1,inplace=True)
        #     print(f2)
            cont_matrix = np.array(f2.loc[:,:])
        #     print(cont_matrix)
            #pro_edge.append(cont_matrix)                        ####get the contact matrix list
        #     print(pro_edge)
            ###########contact features
            ###########
            ###########interaction features(below)
            ##############compute protein AA position and sequence length
            with open(r'%s/%s_protein.pdb'%(input_path,seq_name), "r") as f3:
                mapping_position={}
                Natom=1
                Nresdue=1
                AA={}
                for line in f3:
                    if line.strip().split(' ')[0]=='ATOM':
                        if Natom ==1:
                            AA[Nresdue]=aa_codes[line[17:20]]
                            NAA = line[22:26]
                            chain = line[21:22]
                            mapping_position[NAA+"_"+chain]=Nresdue
                            Natom+=1
                        else:
                            if line[22:26]==NAA:
                                Natom+=1
                            else:
                                NAA=line[22:26]
                                chain = line[21:22]
                                Nresdue+=1
                                AA[Nresdue]=aa_codes[line[17:20]]
                                mapping_position[NAA+"_"+chain]=Nresdue
                                Natom+=1
                protein_length=Nresdue 
            ######
            ######get the normalized position features(1D) and aa list of protein(1D) in interaction files

            with open(r'%s/%s_int.csv'%(out_path,seq_name), "r") as f4:
                interaction_position=[]
                interaction_aa=[]
                interaction_evolutionary=[]
                for line in f4:
                    if line.strip().split(',')[0]!="":
                        Num_Cha=line.strip().split(',')[0][2:]
                        Normalized_position=mapping_position[Num_Cha]/protein_length
                        interaction_position.append(Normalized_position)
                        interaction_aa.append(line.strip().split(',')[0][0])
            ##########
            ######### get the evolutionary features in the interaction   20D
                for j in range(len(interaction_aa)):
                    interaction_evolutionary.append(blosum_dict[pro_aa[j]])
                i_p=[]
                for j in range(len(interaction_aa)):
                    i_p.append([interaction_aa[j]]+[interaction_position[j]]+[interaction_evolutionary[j]])##protein verticle feature in interaction
                #int_provertex.append(i_p)
        #         print(int_provertex)
        ##########
        ########## get the interaction matrix list
            f5=pd.read_csv('%s/%s_int.csv'%(out_path,seq_name))
            column_list=[]
            for column in f5:
                column_list.append(column)
                if (column in list9 or column in list99 or column in listcol99 or column=='H  '):
        #             print(column)
                    int_features=f5.drop(column,axis=1,inplace=True)
        #         if column == 'Unnamed: 0':
        #             f5.drop('Unnamed: 0',axis=1,inplace=True)
            NAME5=f5
            f5=f5.iloc[:,1:]
            Num_atom=len(f5.columns)
            int_matrix = np.array(f5.loc[:,:])
        #     print(seq_name)
            for r in range(int_matrix.shape[0]):
                for l in range(int_matrix.shape[1]):
                    if int_matrix[r][l]!=0:
                        int_matrix[r][l]=(8.0-int_matrix[r][l])/8.0
            #int_edge.append(int_matrix)
        #     print(int_edge)

            ##########
            ########## get ligand vertex features in interaction
        #     ligand_list=[]
            lig_name = mapping_dict[seq_name]
        #     ligand_list.append(lig_name)
        #     for lig in ligand_list:
            atom_list=[]
        #     print(seq_name)
            mol = mol_dict[lig_name]
            for atom in mol.GetAtoms():
        #     print(atom)
        #     print(atom.GetSymbol())
                atom_list.append(atom)
        #     print(atom_list)
            i_l=[]
            for i in range(Num_atom):
                int_ligand=atom_features(atom_list[i])                    ######ligand features in interaction  82D  
                i_l.append(int_ligand)
            #int_ligvertex.append(i_l)   ### the atoms features dim: 82, but it is very sparse !!!!!!!!!!   the above 63 dim may be do some change
        #     print(int_ligvertex)
            ########
            ########
            ######## ligand features in graph
        #     print("10")
            ###### get affinity
        #     aff.append(pid_aff[seq_name])
            fa, fb, anb, bnb, nbs_mat = Mol2Graph(mol)
    #         print(fa)
            if len(fa) == 0:
        #         print(seq_name)
                continue


            

            proid.append(seq_name)
            int_edge.append(int_matrix)
            int_ligvertex.append(i_l)
            int_provertex.append(i_p)
            pro_vertex.append(p)
            pro_edge.append(cont_matrix) 
            mol_inputs.append([fa, fb, anb, bnb, nbs_mat])
            print("seq_name",seq_name)
        except Exception as e:
            print("函数调用出错:")      





    fa_list, fb_list, anb_list, bnb_list, nbs_mat_list = zip(*mol_inputs)
    data_pack = [fa_list, fb_list, anb_list, bnb_list, nbs_mat_list,
                 pro_vertex, pro_edge, int_provertex, int_ligvertex, int_edge,
                 proid]
    # print(data_pack)

    # for i in data_pack:
    #     print(len(i))

    # print(data_pack)

    aa_codes_label = {a:i for i,a in enumerate(aa_codes.values())}
    # print(aa_codes_label)

    def one_hot(i,N):
    #     assert isinstance(i, list)
        arr = np.zeros((len(i),N))  
        arr[tuple(list(zip(*enumerate(i))))]=1
        return arr

    graphs = []
    # graphs_feats = {'id':[],'affinity':[]}

    IT = torch.IntTensor
    FT = torch.FloatTensor

    for s in tqdm(zip(*data_pack),total=len(data_pack[0])):
        try:
            (fa_list, fb_list, anb_list, bnb_list, nbs_mat_list,
            pro_vertex, pro_edge, 
            int_provertex, int_ligvertex, int_edge,
            proid) = s

            mol_node_feats = {'type':[],'feat':[]}
            mol_edges = []
            mol_edges_type = []

            for f in fa_list:
                mol_node_feats['type'].append(f)

            for u,vs in enumerate(anb_list):
                for i,v in enumerate(vs):
                    if not nbs_mat_list[u,i]:
                        break
                    mol_edges.append((u,v))
                    mol_edges_type.append(fb_list[bnb_list[u,i]])
                    mol_edges.append((v,u))
                    mol_edges_type.append(fb_list[bnb_list[u,i]])

            mol_edges = list(zip(*mol_edges))

            ptn_edges = np.where(pro_edge)
            ptn_node_feats = {'type':[],'dis':[],'feat':[]}
            for f in pro_vertex:
        #         print(pro_vertex)
                ptn_node_feats['type'].append(aa_codes_label[f[0]])
                ptn_node_feats['dis'].append(f[1])
                ptn_node_feats['feat'].append(f[2])

            
        #     if proid == '3ag9':
        #         print(int_edge.shape[1], len(mol_node_feats['type']))
        #         print(fa_list, fb_list, anb_list, bnb_list, nbs_mat_list, pro_vertex, pro_edge, int_provertex, int_ligvertex, int_edge, proid,aff)
            assert int_edge.shape[1]==len(mol_node_feats['type'])   ## kaili removed

            for f in int_ligvertex:
                mol_node_feats['feat'].append(f)

            cmb_edges = []
            cmb_edge_weights = []

            dis2id = {v:i for i,v in enumerate(ptn_node_feats['dis'])}
            assert len(dis2id)==len(ptn_node_feats['dis'])
            id2id = {}
            for i,f in enumerate(int_provertex):
                id2id[i]=dis2id[f[1]]

            for pt in range(int_edge.shape[0]):
                for dr in range(int_edge.shape[1]):
                    weight = int_edge[pt,dr]
                    if weight == 0:
                        continue
                    cmb_edges.append((id2id[pt],dr))
                    cmb_edge_weights.append(weight)

            cmb_edges = list(zip(*cmb_edges))

            graph_data = {
                ('atom','a_conn','atom'):tuple([IT(i) for i in mol_edges]),
                ('residue','r_conn','residue'):tuple([IT(i) for i in ptn_edges]),
                ('residue','int_l','atom'):tuple([IT(i) for i in cmb_edges]),
                ('atom','int_r','residue'):tuple([IT(i) for i in cmb_edges][::-1]),
            }
            g = dgl.heterograph(graph_data)
        #     print(g)
            g.edges['a_conn'].data['feat'] = FT(one_hot(mol_edges_type,12))  ##!!!!!! 有6种bond属性，通过组合会有更多的的类别
            g.edges['int'].data['weight'] = torch.Tensor(cmb_edge_weights)
            g.edges['int_r'].data['weight'] = torch.Tensor(cmb_edge_weights)

        #     n_a = g.num_nodes('atom')
        #     for k,v in mol_node_feats.items():
        #         if len(v)!=n_a:
        #             print('atom size not match',proid,len(v),n_a)
        #             v = v[:n_a]
        #         g.nodes['atom'].data[k] = IT(v) if k=='type' else FT(v)

            g.nodes['atom'].data['feat'] = FT(mol_node_feats['feat'])

            residue_feats = FT(
                np.concatenate(
                    (np.expand_dims(ptn_node_feats['dis'],1),
                    ptn_node_feats['feat'],
                    one_hot(ptn_node_feats['type'],20)), axis=-1)
            )
            n_r = g.num_nodes('residue')
            if len(residue_feats)!=n_r:
        #         print('residue size not match',proid,len(residue_feats),n_r)
                for i in nx.components.connected_components(nx.from_numpy_array(pro_edge)):
                    pass
                assert n_r in i and len(i)==1
                residue_feats = residue_feats[:n_r]
            g.nodes['residue'].data['feat'] = residue_feats
        #     print(g)

            graphs.append(g)


        #     graphs_feats['id'].append(proid)
        #     graphs_feats['affinity'].append(aff)
            

            namess.append(proid)
            print(proid)
        except Exception as e:
            print("函数调用出错la") 

    dgl.save_graphs('%s/feats_case.bin'%(out_path),graphs)
    # pd.DataFrame(graphs_feats).to_csv('feats/labels_case.csv',index=False)

def main(input_path,out_path):
    feature_cal(f_g,input_path, out_path, mol_dict, mapping_dict)
 
if __name__ == "__main__":
    main(input_path,out_path)
    print(namess)