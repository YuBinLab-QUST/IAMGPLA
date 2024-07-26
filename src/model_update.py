#the parameter and function renaming file of the train.py file, with the same content

import dgl
import dgl.nn.pytorch as dglnn
import numpy as np
import torch_sparse
import torch
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GATConv,GINConv, global_max_pool,MLP
from torch_geometric.utils import to_dense_adj
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.a_init_v = nn.Linear(82, 120)
        self.a_init_e = nn.Linear(12, 8)

        self.r_init_v1 = nn.Linear(21, 9)
        self.r_init_v2 = nn.Linear(9 + 20, 120)

        self.L = nn.Linear(120, 120)

        self.IIL = Infor_Inter_Layer(120)

        self.PE = Protein_Encoder(120)
        self.LE = Ligand_Encoder(120)

        self.IIE = Inter_Infor_Encoder(1,2,3,120)

        self.classifier = nn.Sequential(
            nn.Linear(120 + 120 + 120 + 6, 298),
            nn.PReLU(),
            nn.Linear(298, 160),
            nn.PReLU(),
            nn.Linear(160, 1)
        )
        self.mean_pool = dglnn.AvgPooling()
        self.sum_pool = dglnn.SumPooling()
    def forward(self, ga, gr, gi, vina,atom_list,res_list): 
        device = torch.device("cuda:0")
        ga = ga.to('cuda:0')
        gr = gr.to('cuda:0')
        gi = gi.to('cuda:0')
        atom_list = atom_list.to('cuda:0')
        res_list = res_list.to('cuda:0') 
        vina = vina.to('cuda:0')

        va_init = self.a_init_v(ga.ndata['feat'])
        ea_init = self.a_init_e(ga.edata['feat'])  
        edge_index_va = ga.adjacency_matrix().indices()

        vr = self.r_init_v1(gr.ndata['feat'][:, :21])    
        vr = torch.cat((vr, gr.ndata['feat'][:, 21:]), -1)
        vr_init = self.r_init_v2(vr) 
        er_init = torch.Tensor().reshape(gr.num_edges(),-1).to(device)
        edge_index_vr = gr.adjacency_matrix().indices()

        vi_a = self.a_init_v(gi.ndata['feat']['atom'])   
        vi_r = self.r_init_v1(gi.ndata['feat']['residue'][:, :21])  
        vi_r = torch.cat((vi_r, gi.ndata['feat']['residue'][:, 21:]), -1)
        vi_r = self.r_init_v2(vi_r)   
        vi_init = torch.cat((vi_a, vi_r), dim=0)
        ei = gi.edata['weight'].reshape(-1)
        ei = torch.cat((ei, ei)).unsqueeze(1) 
        gii = dgl.add_reverse_edges(dgl.to_homogeneous(gi))  
        gii.set_batch_num_nodes(gi.batch_num_nodes('atom') + gi.batch_num_nodes('residue'))  
        gii.set_batch_num_edges(gi.batch_num_edges() * 2)  
        edge_index_vi=gii.adjacency_matrix().indices()

        va = self.IIL(gr, vr_init, ga, va_init)         
        vr = self.IIL(ga, va_init, gr, vr_init)   

        va = va + va_init    
        vr = vr + vr_init

        va = F.leaky_relu(self.L(va), 0.1)  
        sa = self.sum_pool(ga, va)  
        vr = F.leaky_relu(self.L(vr), 0.1)   
        sr = self.sum_pool(gr, vr)  

        GraphDiff = T.GDC(
        self_loop_weight = 1,
        normalization_in = 'sym',
        normalization_out = 'col',
        diffusion_kwargs = dict(method = 'ppr', alpha=0.05),
        sparsification_kwargs = dict(method = 'topk', k = 128, dim = 0),
        exact = True,
        )
        data_a = Data(x=va, edge_index=edge_index_va, num_nodes=ga.num_nodes(), edge_weight=None)
        data_r = Data(x=vr, edge_index=edge_index_vr, num_nodes=gr.num_nodes(), edge_weight=None)

        data_atom = GraphDiff(data_a)
        data_res = GraphDiff(data_r) 

        va = self.LE(ga, va, ea_init, sa, data_atom)  
        vr = self.PE(gr, vr, er_init, sr, data_res)   

        va = va + va_init    
        vr = vr + vr_init  

        fr = self.mean_pool(gr,vr)  
        fa = self.mean_pool(ga,va)   

        vi = self.IIE(vi_init,edge_index_vi,ei,atom_list,res_list)      
        fi = vi

        f = torch.cat((fa, fr, fi, vina), dim=-1)        
        y = self.classifier(f)
        return y
class Infor_Inter_Layer(nn.Module):
    def __init__(self, in_dim):  
        super().__init__()
        self.se = Infor_Inter_Layer.SE_Block(in_dim)
        self.L = nn.Linear(in_dim, 120)
        self.sum_pool = dglnn.SumPooling()     
        self.hw=Infor_Inter_Layer.HighWay(in_dim)
    def forward(self, ga, va, gb, vb):  
        s = self.L(va) 

        with ga.local_scope():    
            ga.ndata['s'] = s
            gga = dgl.unbatch(ga)  
            va = torch.stack([torch.mm(g.ndata['s'].T,self.se(g.ndata['s'])) for g in gga])
            va = va.mean(dim=-1)
        va = dgl.broadcast_nodes(ga, va) 

        vva=torch.sigmoid(s)
        va=torch.mul(vva,va)

        va = self.sum_pool(ga,va)
        va = dgl.broadcast_nodes(gb, va) 
        
        vb = self.L(vb)
        vvb = torch.tanh(vb)
        v = self.hw(va, vvb,vb)
        return v
    
    class SE_Block(nn.Module):
        def __init__(self, in_dim, reduction=8):

            super().__init__()
            self.avg_pool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Sequential(
                nn.Linear(in_dim, in_dim//reduction, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(in_dim//reduction, in_dim, bias=False),
                nn.Sigmoid()
            )

        def forward(self, v):

            vv = v.permute(1, 0)
            vv = vv.unsqueeze(0)
            avg = self.avg_pool(vv).squeeze(2) 
            w = self.fc(avg)
            wv=v*w
            return wv
        
    class HighWay(nn.Module):
        def __init__(self, in_dim):

            super().__init__()
            self.L = nn.Linear(in_dim, in_dim)
            self.gru = nn.GRUCell(in_dim, in_dim)

        def forward(self, a, b, c):

            a=self.L(a)
            b=self.L(b) 
            z = torch.sigmoid(a + b)
            h = z * b + (1 - z) * a
            cc = self.gru(c, h)
            return cc
        
class Protein_Encoder(nn.Module):
    def __init__(self,in_dim, dropout_rate=0.2, bilstm_layers=2, k_head=8):
        super().__init__()
        
        self.is_bidirectional = True
        self.bilstm_layers = bilstm_layers
        self.is_bidirectional = True
        self.lstm = nn.LSTM(in_dim, in_dim, self.bilstm_layers, batch_first=True, bidirectional=self.is_bidirectional, dropout=dropout_rate)
        
        conv = []
        ip = in_dim
        for op in [32, 64,in_dim]:
            conv.append(Protein_Encoder.MLConv(ip, op))
            ip = op
        self.conv = nn.Sequential(*conv)
        
        self.res = Protein_Encoder.Residual(120)
        self.k_head = k_head
        self.att = nn.ModuleList([Protein_Encoder.Attention(in_dim) for _ in range(k_head)])
        
        self.gin = Protein_Encoder.GINs(in_dim)
        
        self.ln = torch.nn.LayerNorm(in_dim * 2)
        self.A = nn.Linear(in_dim* 2,in_dim )
        self.B = nn.Linear(in_dim* k_head, in_dim)
        self.sque = Protein_Encoder.Squeeze()
         
    def forward(self,g, v, e, s, data):

        x = v.unsqueeze(0)
        y = v.unsqueeze(0)
        y = torch.transpose(y, 1, 2)

        v1, _ = self.lstm(x)  # B * tar_len * lstm_dim *2
        v1 = self.ln(v1)
        v1 = self.A(v1)
        v1 = torch.transpose(v1, 1, 2)
        x1 = self.res(v1)
        x1 = x1+y

        v11 = torch.transpose(v1, 1, 2)
        v11 = self.sque(v11)
        v11 = torch.cat([layer(g, v11, s) for layer in self.att],dim=1)
        v11 = torch.tanh(self.B(v11))
        v11 = v11.unsqueeze(0) 
        v11 = torch.transpose(v11, 1, 2)
        
        v2 = self.conv(y)
        v2 = self.res(v2)
        v2 = v2+y
     
        v22 = torch.transpose(v2, 1, 2)
        v22 = self.sque(v22)
        v22 = torch.cat([layer(g, v22, s) for layer in self.att],dim=1)
        v22 = torch.tanh(self.B(v22))
        v22 = v22.unsqueeze(0) 
        v22 = torch.transpose(v22, 1, 2)

        v12 = v2*v11 + x1*v22
        v12 = torch.transpose(v12, 1, 2)
        v12 = self.sque(v12)

        v3 = self.gin(data.x, data.edge_index, data.edge_weight)

        v = v3+v12

        return v

    class GINs(torch.nn.Module):
        def __init__(self, in_dim, hid_dim=128, num_layers=2):
            
            super().__init__()
            out_dim = in_dim
            self.convs = torch.nn.ModuleList()
            for _ in range(num_layers):
                mlp = MLP([in_dim, hid_dim])
                self.convs.append(GINConv(nn=mlp, train_eps=False))
                in_dim = hid_dim
                hid_dim = out_dim

        def forward(self, x, edge_index, edge_weight):
            for conv in self.convs:
                x = conv(x, edge_index).relu()
            return x    
        
    class Attention(nn.Module):
        def __init__(self, in_dim):
            super().__init__()

            self.A = nn.Linear(in_dim, in_dim) 
            self.B = nn.Linear(in_dim, 1)  

        def forward(self, g, v, s):

            d_node = self.A(v)  
            d_super = self.A(s)  
            d_super = dgl.broadcast_nodes(g, d_super)
            g.ndata['a'] = d_super
            w = dgl.softmax_nodes(g, 'a')
            wv = w* d_node 
            g.ndata['h'] = wv
            v = dgl.sum_nodes(g, 'h') 
            v = dgl.broadcast_nodes(g, v)

            return v
    class Squeeze(nn.Module):  
            def forward(self, input: torch.Tensor):
                return input.squeeze()

    class DilatedConv(nn.Module):    
        def __init__(self, nIn, nOut, kSize, stride, d):
            super().__init__()
            padding = int((kSize - 1) / 2) * d
            self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False, dilation=d)

        def forward(self, input):
            output = self.conv(input)
            return output  
    class MLConv(nn.Module):
        def __init__(self, nIn, nOut, add=True):
            super().__init__()
            n = int(nOut / 5)
            n1 = nOut - 4 * n
            self.c1 = nn.Conv1d(nIn, n, 1, padding=0)  
            self.br1 = nn.Sequential(nn.BatchNorm1d(n), nn.PReLU())

            self.d1 = Protein_Encoder. DilatedConv(n, n1, 3, 1, 1)   
            self.d2 = Protein_Encoder. DilatedConv(n, n, 3, 1, 2)   
            self.d4 = Protein_Encoder. DilatedConv(n, n, 3, 1, 4)     
            self.d8 = Protein_Encoder. DilatedConv(n, n, 3, 1, 8)    
            self.d16 = Protein_Encoder. DilatedConv(n, n, 3, 1, 16)   

            self.br2 = nn.Sequential(nn.BatchNorm1d(nOut), nn.PReLU())
            if nIn != nOut:
                add = False
            self.add = add
        def forward(self, v):
            v = self.c1(v)
            v = self.br1(v)
            
            d1 = self.d1(v)
            d2 = self.d2(v)
            d4 = self.d4(v)
            d8 = self.d8(v)
            d16 = self.d16(v)
          
            add1 = d2
            add2 = add1 + d4
            add3 = add2 + d8
            add4 = add3 + d16
            combine = torch.cat([d1, add1, add2, add3, add4], 1)
            if self.add:
                combine = input + combine
            output = self.br2(combine)

            return output
        

    class Residual(nn.Module):

        def __init__(self, in_dim, kernel_size=3, padding=1, stride=1, shortcut=False, downsample=None):
            super().__init__()
            self.conv1 = nn.Conv1d(in_dim, in_dim, kernel_size=kernel_size, padding=padding, stride=stride)
            self.conv2 = nn.Conv1d(in_dim, in_dim, kernel_size=kernel_size, padding=padding, stride=stride)
        def forward(self, x):
            y = F.relu(self.conv1(x))
            y = self.conv2(y)
            return F.relu(x+y)
            
class Ligand_Encoder(nn.Module):
    def __init__(self,in_dim, dropout_rate=0.2,bilstm_layers=2,k_head=8):
        super().__init__()
        
        self.is_bidirectional = True
        self.bilstm_layers = bilstm_layers
        self.is_bidirectional = True
        self.lstm = nn.LSTM(in_dim, in_dim, self.bilstm_layers, batch_first=True,bidirectional=self.is_bidirectional, dropout=dropout_rate)
        
        conv = []
        ip = in_dim
        for op in [32, 64,in_dim]:
            conv.append(Ligand_Encoder.MLConv(ip, op))
            ip = op
        self.conv = nn.Sequential(*conv)
        
        self.res = Ligand_Encoder.Residual(120)
        self.k_head = k_head
        self.att = nn.ModuleList([Ligand_Encoder.Attention(in_dim) for _ in range(k_head)])
        
        self.gin = Ligand_Encoder.GINs(in_dim)
        
        self.ln = torch.nn.LayerNorm(in_dim * 2)
        self.A = nn.Linear(in_dim* 2,in_dim )
        self.B = nn.Linear(in_dim* k_head, in_dim)
        self.sque = Ligand_Encoder.Squeeze()
         
    def forward(self,g, v, e, s, data):

        x = v.unsqueeze(0)
        y = v.unsqueeze(0)
        y = torch.transpose(y, 1, 2)

        v1, _ = self.lstm(x)  # B * tar_len * lstm_dim *2
        v1 = self.ln(v1)
        v1 = self.A(v1)
        v1 = torch.transpose(v1, 1, 2)
        x1 = self.res(v1)
        x1 = x1+y

        v11 = torch.transpose(v1, 1, 2)
        v11 = self.sque(v11)
        v11 = torch.cat([layer(g, v11, s) for layer in self.att],dim=1)
        v11 = torch.tanh(self.B(v11))
        v11 = v11.unsqueeze(0) 
        v11 = torch.transpose(v11, 1, 2)
        
        v2 = self.conv(y)
        v2 = self.res(v2)
        v2 = v2+y
     
        v22 = torch.transpose(v2, 1, 2)
        v22 = self.sque(v22)
        v22 = torch.cat([layer(g, v22, s) for layer in self.att],dim=1)
        v22 = torch.tanh(self.B(v22))
        v22 = v22.unsqueeze(0) 
        v22 = torch.transpose(v22, 1, 2)

        v12 = v2*v11 + x1*v22
        v12 = torch.transpose(v12, 1, 2)
        v12 = self.sque(v12)

        v3 = self.gin(data.x, data.edge_index, data.edge_weight)

        v = v3+v12

        return v


    class GINs(torch.nn.Module):
        def __init__(self, in_dim, hid_dim=128, num_layers=2):
            
            super().__init__()
            out_dim = in_dim
            self.convs = torch.nn.ModuleList()
            for _ in range(num_layers):
                mlp = MLP([in_dim, hid_dim])
                self.convs.append(GINConv(nn=mlp, train_eps=False))
                in_dim = hid_dim
                hid_dim = out_dim

        def forward(self, x, edge_index, edge_weight):
            for conv in self.convs:
                x = conv(x, edge_index).relu()
            return x    
        
    class Attention(nn.Module):
        def __init__(self, in_dim):
            super().__init__()

            self.A = nn.Linear(in_dim, in_dim) 
            self.B = nn.Linear(in_dim, 1)  

        def forward(self, g, v, s):

            d_node = self.A(v)  
            d_super = self.A(s)  
            d_super = dgl.broadcast_nodes(g, d_super)
            g.ndata['a'] = d_super
            w = dgl.softmax_nodes(g, 'a')
            wv = w* d_node 
            g.ndata['h'] = wv
            v = dgl.sum_nodes(g, 'h') 
            v = dgl.broadcast_nodes(g, v)

            return v
    class Squeeze(nn.Module):  
            def forward(self, input: torch.Tensor):
                return input.squeeze()

    class DilatedConv(nn.Module):    
        def __init__(self, nIn, nOut, kSize, stride, d):
            super().__init__()
            padding = int((kSize - 1) / 2) * d
            self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False, dilation=d)

        def forward(self, input):
            output = self.conv(input)
            return output  
    class MLConv(nn.Module):
        def __init__(self, nIn, nOut, add=True):
            super().__init__()
            n = int(nOut / 4)
            n1 = nOut - 3 * n
            self.c1 = nn.Conv1d(nIn, n, 1, padding=0)  
            self.br1 = nn.Sequential(nn.BatchNorm1d(n), nn.PReLU())

            self.d1 = Ligand_Encoder. DilatedConv(n, n1, 3, 1, 1)   
            self.d2 = Ligand_Encoder. DilatedConv(n, n, 3, 1, 2)   
            self.d4 = Ligand_Encoder. DilatedConv(n, n, 3, 1, 4)     
            self.d8 = Ligand_Encoder. DilatedConv(n, n, 3, 1, 8)    

            self.br2 = nn.Sequential(nn.BatchNorm1d(nOut), nn.PReLU())
            if nIn != nOut:
                add = False
            self.add = add
        def forward(self, v):
            v = self.c1(v)
            v = self.br1(v)
            
            d1 = self.d1(v)
            d2 = self.d2(v)
            d4 = self.d4(v)
            d8 = self.d8(v)
          
            add1 = d2
            add2 = add1 + d4
            add3 = add2 + d8

            combine = torch.cat([d1, add1, add2, add3], 1)
            if self.add:
                combine = input + combine
            output = self.br2(combine)

            return output
        

    class Residual(nn.Module):

        def __init__(self, in_dim, kernel_size=3, padding=1, stride=1, shortcut=False, downsample=None):
            super().__init__()
            self.conv1 = nn.Conv1d(in_dim, in_dim, kernel_size=kernel_size, padding=padding, stride=stride)
            self.conv2 = nn.Conv1d(in_dim, in_dim, kernel_size=kernel_size, padding=padding, stride=stride)
        def forward(self, x):
            y = F.relu(self.conv1(x))
            y = self.conv2(y)
            return F.relu(x+y)

class Inter_Infor_Encoder(torch.nn.Module):
    def __init__(self, k1, k2, k3, in_dim, dropout=0.2):
        super().__init__()
       
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3

        self.conv1 = Inter_Infor_Encoder.GAT(in_dim)
        self.conv2 = GCNConv(in_dim, in_dim*2)
        self.conv3 = GCNConv(in_dim*2, in_dim*4)

        self.hw = Inter_Infor_Encoder.HighWay(in_dim)

        self.A = nn.Linear(in_dim,128)
        self.B = nn.Linear(128,120)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, ei, atom_list, res_list):

        inter = [atom_list[i] + res_list[i] for i in range(len(atom_list))]
        batch_list=[]
        for i in range(len(inter)):
            batch_list.extend([i] * inter[i])
        batch_list = np.array(batch_list)
        batch_list = torch.tensor(batch_list)
        batch_list = batch_list.to('cuda:0')
        edge_index = edge_index.to(torch.int64)
        adj = to_dense_adj(edge_index,edge_attr=ei, max_num_nodes=batch_list.shape[0]) 

        if self.k1 == 1:
            h1 = self.conv1(x, edge_index, batch_list)
            h1 = self.relu(h1)
            h1 = self.conv2(h1, edge_index)
            h1 = self.relu(h1)
            h1 = self.conv3(h1, edge_index)
            h1 = self.relu(h1)
        if self.k2 == 2:
            edge_index_square, _ = torch_sparse.spspmm(edge_index, None, edge_index, None, adj.shape[1], adj.shape[1], adj.shape[1], coalesced=True)
            h2 = self.conv1(x,edge_index_square,batch_list)
            h2 = self.relu(h2)
            h2 = self.conv2(h2,edge_index_square)
            h2 = self.relu(h2)
        if self.k3 == 3:
            edge_index_cube, _ = torch_sparse.spspmm(edge_index_square, None, edge_index, None, adj.shape[1], adj.shape[1], adj.shape[1], coalesced=True)
            h3 = self.Conv1(x, edge_index_cube, batch_list)
            h3 = self.relu(h3)

        concat=self.hw(h1,h2, h3)
        vi = global_max_pool(concat,batch_list)
        vi = self.relu(self.A(vi))
        vi = self.dropout(vi)
        vi = self.B(vi)
        vi = self.dropout(vi)
        
        return vi
    class GAT(torch.nn.Module):
        def __init__(self, in_dim, dropout=0.2):
            super().__init__()

 
            self.gat1 = GATConv(in_dim, in_dim, heads=8, dropout=dropout)
            self.gat2 = GATConv(in_dim * 8, in_dim, dropout=dropout)

            self.A = nn.Linear(in_dim, in_dim)
            self.B = nn.Linear(in_dim, 128)
            self.C = nn.Linear(128, in_dim)

            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)

        def forward(self,v, edge_index,batch):

         
            v = F.dropout(v, p=0.2, training=self.training)
            v = F.elu(self.gat1(v, edge_index)) 

            v = F.dropout(v, p=0.2, training=self.training)
            v = self.gat2(v, edge_index)
           
            v = self.relu(v)
            v = self.A(v)
            v = self.relu(v)
            v = self.B(v)
            v = self.relu(v)
            v = self.dropout(v)

            out = self.C(v)
            
            return out
    class HighWay(nn.Module):
        def __init__(self, h_dim):
            super().__init__()
            self.A = nn.Linear(h_dim*4, h_dim)
            self.B = nn.Linear(h_dim*2, h_dim)
            self.gru = nn.GRUCell(h_dim, h_dim)

        def forward(self, a, b, c):

            b=self.B(b)
            a=self.A(a)
            z = torch.sigmoid(a + b)
            h = z * b + (1 - z) * a
            cc = self.gru(c, h)
            return cc