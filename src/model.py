import dgl
import dgl.function as dglfn
import dgl.nn.pytorch as dglnn
import numpy as np
import torch_sparse
import torch
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GATConv, GINConv , global_max_pool ,MLP,global_add_pool
from torch_geometric.utils import to_dense_adj
import itertools#45
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.a_init = nn.Linear(82, 120)
        self.b_init = nn.Linear(12, 8)
        self.r_init_1 = nn.Linear(21, 9)
        self.r_init_2 = nn.Linear(9 + 20, 120)
        self.A = nn.Linear(120, 120)
        self.C = nn.Linear(120, 120)
        # message transport
        self.r_mt = MT(120)
        self.a_mt = MT(120)
        self.aa = A(120)
        self.bb = B(120)
        # ligand
        # interaction
        self.i_conf = GCNNet(1,2,3,120)
        # predict
        self.classifier = nn.Sequential(
            nn.Linear(120 + 120 + 120 + 6, 298),
            nn.PReLU(),
            nn.Linear(298, 160),
            nn.PReLU(),
            nn.Linear(160, 1)
        )
        self.sum_pool = dglnn.SumPooling()
        self.mean_pool = dglnn.AvgPooling()
        self.max_pool = dglnn.MaxPooling()
    def forward(self, ga, gr, gi, vina,atom_list,res_list): 
        device = torch.device("cuda:0")
        #print(vina.shape)#torch.Size([8, 6])
        ga = ga.to('cuda:0')
        gr = gr.to('cuda:0')
        gi = gi.to('cuda:0')
        atom_list = atom_list.to('cuda:0')
        res_list = res_list.to('cuda:0') 
        vina = vina.to('cuda:0')


        """ import pickle
        datas=str(ga.ndata['feat'].tolist())
        with open("/opt/data/private/pdb20_8/eva/atom.pkl", "wb") as f:
            pickle.dump(datas, f) 
        print("ok")  """


        print("11111")




        va_init = self.a_init(ga.ndata['feat'])#num_atom,120
        #print("va_init",va_init.shape) 
        ea = self.b_init(ga.edata['feat'])  #num_atom_edge,8
        #print("ea",ea.shape) 
        adj_va=ga.adjacency_matrix().indices()
        vr = self.r_init_1(gr.ndata['feat'][:, :21])    
        vr = torch.cat((vr, gr.ndata['feat'][:, 21:]), -1)
        vr_init = self.r_init_2(vr) 
        adj_vr=gr.adjacency_matrix().indices()
        #print("vr_init",vr_init.shape) #num_res,120
        vi_a = self.a_init(gi.ndata['feat']['atom'])   
        vi_r = self.r_init_1(gi.ndata['feat']['residue'][:, :21])  
        vi_r = torch.cat((vi_r, gi.ndata['feat']['residue'][:, 21:]), -1)
        vi_r = self.r_init_2(vi_r)   
        vi_init = torch.cat((vi_a, vi_r), dim=0) #num(atom+res),120
        er=torch.Tensor().reshape(gr.num_edges(),-1).to(device)
        #print("vi_init",vi_init.shape)
        ei = gi.edata['weight'].reshape(-1)
        ei = torch.cat((ei, ei)).unsqueeze(1)  #向连边*2 ，1
        #print("ei",ei.shape)
        gii = dgl.add_reverse_edges(dgl.to_homogeneous(gi))  
        gii.set_batch_num_nodes(gi.batch_num_nodes('atom') + gi.batch_num_nodes('residue'))  
        gii.set_batch_num_edges(gi.batch_num_edges() * 2)  
        adj_vi=gii.adjacency_matrix().indices()

        

        va = self.a_mt(gr, vr_init, ga, va_init)         
        vr = self.r_mt(ga, va_init, gr, vr_init)   
        va = va + va_init    
        vr = vr + vr_init
        va = F.leaky_relu(self.A(va), 0.1)  
        sa = self.sum_pool(ga, va)  

        vr = F.leaky_relu(self.C(vr), 0.1)   
        sr = self.sum_pool(gr, vr)  



        transform = T.GDC(
        self_loop_weight=1,
        normalization_in='sym',
        normalization_out='col',
        diffusion_kwargs=dict(method='ppr', alpha=0.05),
        sparsification_kwargs=dict(method='topk', k=128, dim=0),
        exact=True,
        )

        data_a= Data(x=va, edge_index=adj_va,num_nodes=ga.num_nodes() ,edge_weight= None)
        data_r= Data(x=vr, edge_index=adj_vr,num_nodes=gr.num_nodes() ,edge_weight= None)


        data_atom= transform(data_a)
        data_res= transform(data_r) 

        va = self.bb(ga, va, ea, sa,data_atom)  
        vr = self.aa(gr, vr, er, sr,data_res)   

        va = va + va_init    
        vr =  vr+vr_init   
        fr = self.mean_pool(gr,vr)  
        fa = self.mean_pool(ga,va)   
        # interaction
        vi = self.i_conf(vi_init,adj_vi,ei,atom_list,res_list)      
        """ print(vi.shape)
        print(gii) """
        fi = vi
        f = torch.cat((fa, fr, fi, vina), dim=-1)        
        y = self.classifier(f)
        return y
class MT(nn.Module):

    def __init__(self, in_dim):  
        super().__init__()

        """ self.A = nn.Linear(in_dim, 64) 
        
         """
        self.A = MT.se_attention(in_dim)
        self.B = nn.Linear(in_dim, 120)
        self.C = nn.Linear(in_dim, 8)
        self.sum_pool = dglnn.SumPooling()
         
        self.E = nn.Linear(in_dim, 120)
        self.F=MT.GateUpdate(in_dim)
    def forward(self, ga, va, gb, vb):  
        s = self.B(va) 
        
        with ga.local_scope():    
            ga.ndata['s'] = s
            gga = dgl.unbatch(ga)  
            gp_ = torch.stack([torch.mm(g.ndata['s'].T,self.A(g.ndata['s'])) for g in gga])
            gp_ = gp_.mean(dim=-1)
        
        gp_ = dgl.broadcast_nodes(ga, gp_) 
  
        va1=torch.sigmoid(s)
        gp_=torch.mul(va1,gp_)
        gp_=self.sum_pool(ga,gp_)
          
        gp_ = dgl.broadcast_nodes(gb, gp_) 
       
        vb = self.E(vb)
        vb1=torch.tanh(vb)
        #print(vb.shape)
        vbb = self.F(gp_, vb1,vb)
        return vbb
    class se_attention(nn.Module):
        def __init__(self, inputs, reduction=8):
            super().__init__()
        
            self.avg_pool = nn.AdaptiveAvgPool1d(1)

            self.fc = nn.Sequential(
                nn.Linear(inputs, inputs//reduction, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(inputs//reduction, inputs, bias=False),
                nn.Sigmoid()
            )
        def forward(self, x):
            x1=x.permute(1, 0)
            x2 = x1.unsqueeze(0)
            avg = self.avg_pool(x2).squeeze(2) 
            att = self.fc(avg)
            y=x*att
            return y
    class GateUpdate(nn.Module):
        def __init__(self, h_dim):
            super().__init__()
            self.A = nn.Linear(h_dim, h_dim)
            self.B = nn.Linear(h_dim, h_dim)
            self.gru = nn.GRUCell(h_dim, h_dim)

        def forward(self, a, b, c):
            """ print("a",a.shape)
            self.B(b)
            print("b",b.shape)
            print("c",c.shape)  """
            b=self.B(b)
            a=self.A(a)
            z = torch.sigmoid(a + b)
        
            """ print(x.shape)
            print(x.size(1)) """
            #print("z",z.shape)
            h = z * b + (1 - z) * a
           
            #print("h",h.shape)
            cc = self.gru(c, h)

          
            return cc

class A(nn.Module):
    def __init__(self,embed_size, dropout_rate=0.2,bilstm_layers=2,k_head=8):
        super().__init__()
        self.is_bidirectional = True
        self.dropout = nn.Dropout(dropout_rate)
        self.bilstm_layers = bilstm_layers
        self.is_bidirectional = True
        #self.smiles_input_fc = nn.Linear(256, lstm_dim)
        self.lstm = nn.LSTM(embed_size, embed_size, self.bilstm_layers, batch_first=True,
                                  bidirectional=self.is_bidirectional, dropout=dropout_rate)
        self.ln1 = torch.nn.LayerNorm(embed_size * 2)
        self.fc = nn.Linear(embed_size* 2,embed_size )
        self.res=A.BasicConvResBlock(120)
        # DilatedConv Module
        conv_seq = []
        ic = embed_size
        for oc in [32, 64,embed_size]:
            conv_seq.append(A.DilatedConvBlock(ic, oc))
            ic = oc
        """ conv_seq.append(nn.AdaptiveMaxPool1d(1))
        conv_seq.append(CAPLA.Squeeze()) """
        self.conv_seq = nn.Sequential(*conv_seq)
        self.sque=A.Squeeze()
        self.k_head=k_head
        self.B = nn.Linear(embed_size* k_head, embed_size)
        self.m2s = nn.ModuleList([A.Helper(embed_size, embed_size) for _ in range(k_head)])
        self.cngs=A.GIN(embed_size)
    def forward(self,g, v, e, s,data):
        x = v.unsqueeze(0)
        y = v.unsqueeze(0)
        y = torch.transpose(y, 1, 2)

        x1, _ = self.lstm(x)  # B * tar_len * lstm_dim *2
        x1 = self.ln1(x1)
        #print("ln1",data.shape)
        x1=self.fc(x1)
        #print("fc",data.shape)
        x1  = torch.transpose(x1, 1, 2)
        
        #print("trans",data.shape)
        #data=y+x
        v1=self.res(x1)
        #print(data.shape)
        v1=v1+y

        output_1  = torch.transpose(x1, 1, 2)
        v11=self.sque(output_1)
        m2s1 = torch.cat([layer(g, v11, s) for layer in self.m2s],dim=1)
        m2s1 = torch.tanh(self.B(m2s1))

        m2s1= m2s1.unsqueeze(0) 
        m2s1  = torch.transpose(m2s1, 1, 2)
        

        seq_conv2 = self.conv_seq(y)
        seq_conv2=self.res(seq_conv2)
        seq_conv2=seq_conv2+y
     

        output_2  = torch.transpose(seq_conv2, 1, 2)
        v2=self.sque(output_2)
        m2s2 = torch.cat([layer(g, v2, s) for layer in self.m2s],dim=1)
        #print("m2s",m2s.shape)
        m2s2 = torch.tanh(self.B(m2s2))
        m2s2= m2s2.unsqueeze(0) 
        m2s2  = torch.transpose(m2s2, 1, 2)


        optput=seq_conv2*m2s1+v1*m2s2
  
        optput = torch.transpose(optput, 1, 2)
        optput=self.sque(optput)
        #print(seq_conv.shape)


        xxx=self.cngs(data.x, data.edge_index, data.edge_weight)
        optput=xxx+optput

        return optput

    class GIN(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels=128, num_layers=2):
            super().__init__()
            out_channels=in_channels
            self.convs = torch.nn.ModuleList()
            for _ in range(num_layers):
                mlp = MLP([in_channels, hidden_channels])
                self.convs.append(GINConv(nn=mlp, train_eps=False))
                in_channels = hidden_channels
                hidden_channels=out_channels
        def forward(self, x, edge_index, edge_weight):
            for conv in self.convs:
                x = conv(x, edge_index).relu()
            # Pass the batch size to avoid CPU communication/graph breaks:
           
            return x    

    class Helper(nn.Module):
        def __init__(self, v_dim, h_dim):
            super().__init__()

            self.A = nn.Linear(v_dim, h_dim) 
            self.B = nn.Linear(v_dim, h_dim) 
            self.C = nn.Linear(h_dim, 1)  
            self.D = nn.Linear(v_dim, h_dim) 

        def forward(self, g, v, s):
           # print(v.shape)
            d_node = self.A(v)  #1000 120
            d_super = self.B(s)  #8 120
            d_super = dgl.broadcast_nodes(g, d_super)
            #print(d_super.shape)
            g.ndata['a'] = d_super
            a = dgl.softmax_nodes(g, 'a')
            aa = a* d_node  #1000
            #print("aa",aa.shape)
            g.ndata['h'] = aa
            h = dgl.sum_nodes(g, 'h') 
            h=  dgl.broadcast_nodes(g, h)
            #print("h",h.shape)

            return h
    class Squeeze(nn.Module):   #Dimention Module
            def forward(self, input: torch.Tensor):
                return input.squeeze()

    class DilatedConv(nn.Module):     # Dilated Convolution
        def __init__(self, nIn, nOut, kSize,stride,d ):
            super().__init__()
            padding = int((kSize - 1) / 2) * d
            self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False, dilation=d)

        def forward(self, input):
            output = self.conv(input)
            return output  
    class DilatedConvBlock(nn.Module):
        def __init__(self, nIn, nOut, add=True):
            super().__init__()
            n = int(nOut / 5)
            n1 = nOut - 4 * n
            self.c1 = nn.Conv1d(nIn, n, 1, padding=0)  # Down Dizzzmention
            self.br1 = nn.Sequential(nn.BatchNorm1d(n), nn.PReLU())
            self.d1 =A. DilatedConv(n, n1, 3, 1, 1)    # Dilated scale:1(2^0)
            self.d2 =A. DilatedConv(n, n, 3, 1, 2)     # Dilated scale:2(2^1)
            self.d4 =A. DilatedConv(n, n, 3, 1, 4)     # Dilated scale:4(2^2)
            self.d8 = A.DilatedConv(n, n, 3, 1, 8)     # Dilated scale:8(2^3)
            self.d16 =A. DilatedConv(n, n, 3, 1, 16)   # Dilated scale:16(2^4)
            self.br2 = nn.Sequential(nn.BatchNorm1d(nOut), nn.PReLU())

            if nIn != nOut:
                add = False
            self.add = add
        def forward(self, input):
            output1 = self.c1(input)
            output1 = self.br1(output1)
            d1 = self.d1(output1)
           
            d2 = self.d2(output1)
            d4 = self.d4(output1)
            d8 = self.d8(output1)
            d16 = self.d16(output1)
          
            add1 = d2
            add2 = add1 + d4
            add3 = add2 + d8
            add4 = add3 + d16
            combine = torch.cat([d1, add1, add2, add3, add4], 1)
            if self.add:
                combine = input + combine
            output = self.br2(combine)

            return output
        

    class BasicConvResBlock(nn.Module):

        def __init__(self, input_dim, out_channels=120, kernel_size=3, padding=1, stride=1, shortcut=False, downsample=None):
            super().__init__()
            self.conv1 = nn.Conv1d(input_dim, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
            self.conv2 = nn.Conv1d(out_channels, input_dim, kernel_size=kernel_size, padding=padding, stride=stride)
        def forward(self, x):
            y = F.relu(self.conv1(x))
            y = self.conv2(y)
            return F.relu(x+y)
            
class B(nn.Module):
    def __init__(self,embed_size, dropout_rate=0.2,bilstm_layers=2,k_head=8):
        super().__init__()
        self.is_bidirectional = True
        self.dropout = nn.Dropout(dropout_rate)
        self.bilstm_layers = bilstm_layers
        self.is_bidirectional = True
        #self.smiles_input_fc = nn.Linear(256, lstm_dim)
        self.lstm = nn.LSTM(embed_size, embed_size, self.bilstm_layers, batch_first=True,
                                  bidirectional=self.is_bidirectional, dropout=dropout_rate)
        self.ln1 = torch.nn.LayerNorm(embed_size * 2)
        self.fc = nn.Linear(embed_size* 2,embed_size )
        self.res=B.BasicConvResBlock(120)
        # DilatedConv Module
        conv_seq = []
        ic = embed_size
        for oc in [32, 64,embed_size]:
            conv_seq.append(B.DilatedConvBlock(ic, oc))
            ic = oc
        """ conv_seq.append(nn.AdaptiveMaxPool1d(1))
        conv_seq.append(CAPLA.Squeeze()) """
        self.conv_seq = nn.Sequential(*conv_seq)
        self.sque=B.Squeeze()
        self.k_head=k_head
        self.B = nn.Linear(embed_size* k_head, embed_size)
        self.m2s = nn.ModuleList([B.Helper(embed_size, embed_size) for _ in range(k_head)])
        self.cngs=A.GIN(embed_size)
    def forward(self,g, v, e, s,data):
        x = v.unsqueeze(0)
        y = v.unsqueeze(0)
        y = torch.transpose(y, 1, 2)

        x1, _ = self.lstm(x)  # B * tar_len * lstm_dim *2
        x1 = self.ln1(x1)
        #print("ln1",data.shape)
        x1=self.fc(x1)
        #print("fc",data.shape)
        x1  = torch.transpose(x1, 1, 2)
        
        #print("trans",data.shape)
        #data=y+x
        v1=self.res(x1)
        #print(data.shape)
        v1=v1+y

        output_1  = torch.transpose(x1, 1, 2)
        v11=self.sque(output_1)
        m2s1 = torch.cat([layer(g, v11, s) for layer in self.m2s],dim=1)
        m2s1 = torch.tanh(self.B(m2s1))

        m2s1= m2s1.unsqueeze(0) 
        m2s1  = torch.transpose(m2s1, 1, 2)
        

        seq_conv2 = self.conv_seq(y)
        seq_conv2=self.res(seq_conv2)
        seq_conv2=seq_conv2+y
     

        output_2  = torch.transpose(seq_conv2, 1, 2)
        v2=self.sque(output_2)
        m2s2 = torch.cat([layer(g, v2, s) for layer in self.m2s],dim=1)
        #print("m2s",m2s.shape)
        m2s2 = torch.tanh(self.B(m2s2))
        m2s2= m2s2.unsqueeze(0) 
        m2s2  = torch.transpose(m2s2, 1, 2)


        optput=seq_conv2*m2s1+v1*m2s2
  
        optput = torch.transpose(optput, 1, 2)
        optput=self.sque(optput)

        xxx=self.cngs(data.x, data.edge_index, data.edge_weight)
        optput=xxx+optput

        return optput


    class GIN(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels=128, num_layers=2):
            super().__init__()
            out_channels=in_channels
            self.convs = torch.nn.ModuleList()
            for _ in range(num_layers):
                mlp = MLP([in_channels, hidden_channels])
                self.convs.append(GINConv(nn=mlp, train_eps=False))
                in_channels = hidden_channels
                hidden_channels=out_channels
        def forward(self, x, edge_index, edge_weight):
            for conv in self.convs:
                x = conv(x, edge_index).relu()
            # Pass the batch size to avoid CPU communication/graph breaks:
           
            return x    
    class Helper(nn.Module):
        def __init__(self, v_dim, h_dim):
            super().__init__()

            self.A = nn.Linear(v_dim, h_dim) 
            self.B = nn.Linear(v_dim, h_dim) 
            self.C = nn.Linear(h_dim, 1)  
            self.D = nn.Linear(v_dim, h_dim) 

        def forward(self, g, v, s):
           # print(v.shape)
            d_node = self.A(v)  #1000 120
            d_super = self.B(s)  #8 120
            d_super = dgl.broadcast_nodes(g, d_super)
            #print(d_super.shape)
            g.ndata['a'] = d_super
            a = dgl.softmax_nodes(g, 'a')
            aa = a* d_node  #1000
            #print("aa",aa.shape)
            g.ndata['h'] = aa
            h = dgl.sum_nodes(g, 'h') 
            h=  dgl.broadcast_nodes(g, h)
            #print("h",h.shape)

            return h
    class Squeeze(nn.Module):   #Dimention Module
            def forward(self, input: torch.Tensor):
                return input.squeeze()

    class DilatedConv(nn.Module):     # Dilated Convolution
        def __init__(self, nIn, nOut, kSize, stride, d):
            super().__init__()
            padding = int((kSize - 1) / 2) * d
            self.conv = nn.Conv1d(nIn, nOut, kSize,stride=stride, padding=padding, bias=False, dilation=d)

        def forward(self, input):
            output = self.conv(input)
            return output  
    class DilatedConvBlock(nn.Module):
        def __init__(self, nIn, nOut, add=True):
            super().__init__()
            n = int(nOut / 4)
            n1 = nOut - 3 * n
            self.c1 = nn.Conv1d(nIn, n, 1, padding=0)
            self.br1 = nn.Sequential(nn.BatchNorm1d(n), nn.PReLU())
            self.d1 =   B.DilatedConv(n, n1, 3, 1, 1)  # Dilated scale:1(2^0)
            self.d2 = B.DilatedConv(n, n, 3, 1, 2)   # Dilated scale:2(2^1)
            self.d4 = B.DilatedConv(n, n, 3, 1, 4)   # Dilated scale:4(2^2)
            self.d8 = B.DilatedConv(n, n, 3, 1, 8)   # Dilated scale:8(2^3)
            self.br2 = nn.Sequential(nn.BatchNorm1d(nOut), nn.PReLU())

            if nIn != nOut:
                add = False
            self.add = add

        def forward(self, input):

            output1 = self.c1(input)
            output1 = self.br1(output1)
            d1 = self.d1(output1)
            d2 = self.d2(output1)
            d4 = self.d4(output1)
            d8 = self.d8(output1)

            add1 = d2
            add2 = add1 + d4
            add3 = add2 + d8

            combine = torch.cat([d1, add1, add2, add3], 1)

            if self.add:
                combine = input + combine
            output = self.br2(combine)
            return output  
    class BasicConvResBlock(nn.Module):

        def __init__(self, input_dim, out_channels=120, kernel_size=3, padding=1, stride=1, shortcut=False, downsample=None):
            super().__init__()
            self.conv1 = nn.Conv1d(input_dim, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
            self.conv2 = nn.Conv1d(out_channels, input_dim, kernel_size=kernel_size, padding=padding, stride=stride)
        def forward(self, x):
            y = F.relu(self.conv1(x))
            y = self.conv2(y)
            return F.relu(x+y)

class GCNNet(torch.nn.Module):
    def __init__(self,k1,k2,k3,num_feature_xd,dropout=0.2):
        super().__init__()
       
        # Smile graph branch
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.embed_dim = num_feature_xd
  
        self.num_feature_xd=num_feature_xd
        self.Conv1 = GCNNet.GAT(num_feature_xd)
        self.Conv2 = GCNConv(num_feature_xd,num_feature_xd*2)
        self.Conv3 = GCNConv(num_feature_xd*2,num_feature_xd*4)
        self.relu = nn.ReLU()
        self.leakrelu = nn.LeakyReLU()
        self.fc_g1 = nn.Linear(num_feature_xd,128)
        self.fc_g2 = nn.Linear(128,120)
        #self.fc_g3 = nn.Linear(360,120)
        self.dropout = nn.Dropout(dropout)
        self.gru = GCNNet.GateUpdate(num_feature_xd)
    def forward(self,x,edge_index,ei,atom_list,res_list):
        inter = [atom_list[i] + res_list[i] for i in range(len(atom_list))]

        batch_list=[]
        for i in range(len(inter)):
            batch_list.extend([i] * inter[i])
        batch_list = np.array(batch_list)
        batch_list = torch.tensor(batch_list)
        batch_list = batch_list.to('cuda:0')
        """ print( batch_list.shape) 
        print(x.shape)#5160 120(
        print(edge_index.shape)#[2, 6982])
        print(ei.shape)#[6982, 1]) """
        
        #print(x.shape)#torch.Size([3524, 120])
        edge_index = edge_index.to(torch.int64)
        #print(edge_index.shape)
        adj = to_dense_adj(edge_index,edge_attr=ei, max_num_nodes=batch_list.shape[0]) 
        #print(adj.shape)
        """ print(edge_index.shape)#2 6982
        print(adj.shape)#torch.Size([1, 5153, 5153, 1]) """
        if self.k1 == 1:
            h1 = self.Conv1(x,edge_index,batch_list)
            h1 = self.relu(h1)
            h2 = self.Conv2(h1,edge_index)
            h2 = self.relu(h2)
            h3 = self.Conv3(h2,edge_index)
            h3 = self.relu(h3)

        #print(h3.shape)#torch.Size([3524, 480])
        if self.k2 == 2:
            edge_index_square,_ = torch_sparse.spspmm(edge_index,None,edge_index,None,adj.shape[1],adj.shape[1],adj.shape[1],coalesced=True)
            h4 = self.Conv1(x,edge_index_square,batch_list)
            h4 = self.relu(h4)
            h5 = self.Conv2(h4,edge_index_square)
            h5 = self.relu(h5)

        #print(h5.shape)#torch.Size([3524, 240])
        if self.k3 == 3:
            edge_index_cube,_ = torch_sparse.spspmm(edge_index_square,None,edge_index,None,adj.shape[1],adj.shape[1],adj.shape[1],coalesced=True)
            h6 = self.Conv1(x,edge_index_cube,batch_list)
            h6 = self.relu(h6)
        
        concat=self.gru(h3,h5, h6)
        #concat = torch.cat([h3,h5,h6],dim=1) 

        x = global_max_pool(concat,batch_list)#global_max_pooling
        """ batch_ = torch.zeros(x.shape[0])
     
        batch_= batch_.to('cuda:0')
        batch_= batch_.to(torch.int64) """
        
        #print(x.shape) #8*840
        #flatten
    #flatten
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
    
        x = self.fc_g2(x)
        x = self.dropout(x)
        

        return x
    class GAT(torch.nn.Module):
        def __init__(self, num_features_xd, dropout=0.2):
            super().__init__()

            # graph layers
            self.gcn1 = GATConv(num_features_xd, num_features_xd, heads=8, dropout=dropout)
            self.gcn2 = GATConv(num_features_xd * 8, num_features_xd, dropout=dropout)

            self.fc_g1 = nn.Linear(num_features_xd, num_features_xd)

            # combined layers
            self.fc1 = nn.Linear(num_features_xd,128)
            
            self.out = nn.Linear(128, num_features_xd)

            # activation and regularization
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)

        def forward(self,x, edge_index,batch):
            # graph input feed-forward
             #4622 120 
         
            x = F.dropout(x, p=0.2, training=self.training)
  
            x = F.elu(self.gcn1(x, edge_index)) #4622 120 

            x = F.dropout(x, p=0.2, training=self.training)
            x = self.gcn2(x, edge_index)
           
            x = self.relu(x)
            #x = global_max_pool(x, batch)  # global max pooling
           
            x = self.fc_g1(x)
            x = self.relu(x)

            # add some dense layers
            xc = x
            xc = self.fc1(xc)
            xc = self.relu(xc)
            xc = self.dropout(xc)

            out = self.out(xc)
            
            return out
    class GateUpdate(nn.Module):
        def __init__(self, h_dim):
            super().__init__()
            self.A = nn.Linear(h_dim*4, h_dim)
            self.B = nn.Linear(h_dim*2, h_dim)
            self.gru = nn.GRUCell(h_dim, h_dim)

        def forward(self, a, b, c):
            """ print("a",a.shape)
            self.B(b)
            print("b",b.shape)
            print("c",c.shape)  """
            b=self.B(b)
            a=self.A(a)
            z = torch.sigmoid(a + b)
        
            """ print(x.shape)
            print(x.size(1)) """
            #print("z",z.shape)
            h = z * b + (1 - z) * a
           
            #print("h",h.shape)
            cc = self.gru(c, h)

          
            return cc