import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import pickle
import dgl
from tqdm.auto import tqdm
from dataset import SepDataset, collate_fn
from model import ModelNew
import metrics

def test(model, valid_loader_compound, criterion,device):
    model.eval()
    losses = []
    outputs = []
    targets = []
    name_list =[]
    tbar = tqdm(valid_loader_compound, total=len(valid_loader_compound))
    for i, data in enumerate(tbar):
        data0 = [i.to(device) for i in data[0]]
        ga, gr, gi, aff = data0
        vina = data[1]
        idnames = data[2]
        atom_list=data[3]
        res_list=data[4]
        name_l = []
        for name in idnames:
            name_l.append(name)
        name_list.append(name_l)
        with torch.no_grad():
            y_pred = model(ga,gr,gi,vina,atom_list,res_list).squeeze()
        y_true = aff.float().squeeze()
        assert y_pred.shape == y_true.shape
        loss = criterion(y_pred,y_true).cuda() 
        losses.append(loss.item())
        outputs.append(y_pred.cpu().detach().numpy().reshape(-1))
        targets.append(y_true.cpu().detach().numpy().reshape(-1))
    targets = np.concatenate(targets).reshape(-1)
    outputs = np.concatenate(outputs).reshape(-1)
    name_list = np.concatenate(name_list).reshape(-1)
    evaluation = {
        'c_index': metrics.c_index(targets, outputs),
        'RMSE': metrics.RMSE(targets, outputs),
        'MAE': metrics.MAE(targets, outputs),
        'SD': metrics.SD(targets, outputs),
        'CORR': metrics.CORR(targets, outputs),}
    
    return evaluation,targets, outputs, name_list

def main():
    flag = 'predict' 
    #model_path = 'model/model.pth' 
    model_path = 'model/model_finally.pth' 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SHOW_PROCESS_BAR = False
    
    vina_list = []
    graphs = dgl.load_graphs('data/graph_16_8.bin')[0]
    labels = pd.read_csv('data/aff_16_8.csv')
    vina_terms =open(r'data/vina_16_8.pkl','rb')
    vina = pickle.load(vina_terms)
    for i in range(279):# number
        if labels.id[i] in vina.keys():
            vina_list.append(vina[labels.id[i]])

    test_dataset = SepDataset([graphs[i] for i in range(len(graphs))], [vina_list[i] for i in range(len(vina_list))], [labels.id[i] for i in range(len(labels))], [labels.affinity[i] for i in range(len(labels))], ['a_conn','r_conn', 'int_l'])
    test20_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0, collate_fn=collate_fn)

    model = ModelNew()   
    checkpoint = torch.load(model_path,map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    criterion = torch.nn.MSELoss()
    p = test20_loader
    p_f = 'test'   
    print(f'{flag}_{p_f}.csv')
    evoluation,targets,outputs,names = test(model, p, criterion,device)
    a = pd.DataFrame()
    a=a.assign(pdbid=names,predicted=outputs,real=targets,set=p_f)
    a.to_csv(f'result/{flag}_{p_f}.csv')  
    

    print(evoluation)
if __name__ == "__main__":
    main()