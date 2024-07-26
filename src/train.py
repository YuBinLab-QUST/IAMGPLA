import pickle
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import dgl          
from tqdm.auto import tqdm
from IPython.display import clear_output
from dataset import SepDataset, collate_fn
from model import ModelNew
import metrics
import datetime
import pytz

current_time = datetime.datetime.now()
timezone = pytz.timezone('Asia/Shanghai')
current_time = current_time.astimezone(timezone)
time_string = current_time.strftime("%Y-%m-%d_%H-%M-%S")

seed = np.random.randint(2021, 2022) ##random
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)
np.random.seed(seed)


def train(model, train_loader_compound, criterion, optimizer,epoch,device):
    model.train()
    tbar = tqdm(train_loader_compound, total=len(train_loader_compound))
    losses = []
    outputs = []
    targets = []
    for i, data in enumerate(tbar):
        data0 = [i.to(device) for i in data[0]]
        ga, gr, gi, aff = data0 
        atom_list=data[3]
        res_list=data[4]
        vina = data[1]
        y_pred = model(ga,gr,gi,vina,atom_list,res_list).squeeze()
        y_true = aff.float().squeeze()
        
        assert y_pred.shape == y_true.shape
        loss = criterion(y_pred,y_true).cuda()
        loss.backward()
        optimizer.step()    
        optimizer.zero_grad()     
        losses.append(loss.item())
        outputs.append(y_pred.cpu().detach().numpy().reshape(-1))
        targets.append(y_true.cpu().detach().numpy().reshape(-1))

    targets = np.concatenate(targets).reshape(-1)
    outputs = np.concatenate(outputs).reshape(-1)
        
    evaluation = {
        'c_index': metrics.c_index(targets, outputs),
        'RMSE': metrics.RMSE(targets, outputs),
        'MAE': metrics.MAE(targets, outputs),
        'SD': metrics.SD(targets, outputs),
        'CORR': metrics.CORR(targets, outputs),}    
    m_losses=np.mean(losses)
    
    return m_losses,evaluation

def valid(model, valid_loader_compound, criterion,device):
    model.eval()
    losses = []
    outputs = []
    targets = []
    tbar = tqdm(valid_loader_compound, total=len(valid_loader_compound))
    for i, data in enumerate(tbar):
        data0 = [i.to(device) for i in data[0]]
        ga, gr, gi, aff = data0 
        atom_list=data[3]
        res_list=data[4]
        vina = data[1]
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
        
    evaluation = {
        'c_index': metrics.c_index(targets, outputs),
        'RMSE': metrics.RMSE(targets, outputs),
        'MAE': metrics.MAE(targets, outputs),
        'SD': metrics.SD(targets, outputs),
        'CORR': metrics.CORR(targets, outputs),}
    ml=np.mean(losses)  
    
    return ml, evaluation

def main():
    F=open(r'data/data_8.pkl','rb')
    content=pickle.load(F)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    graphs = dgl.load_graphs('data/graph_20_8.bin')[0]
    labels = pd.read_csv('data/aff_20_8.csv')
    vina_terms=open(r'data/vina_20_8.pkl','rb')
    vina=pickle.load(vina_terms)
    vina_list= []
    for i in range(17381):
        if labels.id[i] in vina.keys():
            vina_list.append(vina[labels.id[i]])

    
    compound_train = content[0]
    compound_valid = content[1]
    compound_test = content[2]
    
    train_dataset_compound = SepDataset([graphs[i] for i in compound_train], [vina_list[i] for i in compound_train], [labels.id[i] for i in compound_train], [labels.affinity[i] for i in compound_train], ['a_conn','r_conn', 'int_l'])
    valid_dataset_compound = SepDataset([graphs[i] for i in compound_valid], [vina_list[i] for i in compound_valid], [labels.id[i] for i in compound_valid], [labels.affinity[i] for i in compound_valid], ['a_conn','r_conn', 'int_l'])
    test_dataset_compound = SepDataset([graphs[i] for i in compound_test], [vina_list[i] for i in compound_test], [labels.id[i] for i in compound_test], [labels.affinity[i] for i in compound_test], ['a_conn','r_conn', 'int_l']) 
    
    train_loader_compound = DataLoader(train_dataset_compound, batch_size=8, shuffle=True, num_workers=8, collate_fn=collate_fn,pin_memory=False,drop_last=False,)
    valid_loader_compound = DataLoader(valid_dataset_compound, batch_size=8, shuffle=False, num_workers=8, collate_fn=collate_fn)
    test_loader_compound = DataLoader(test_dataset_compound, batch_size=8, shuffle=False, num_workers=8, collate_fn=collate_fn)

    model = ModelNew()
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), 1.2e-4, weight_decay=1e-6)   ### (model.parameters(), 1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=40, eta_min=1e-6)
    criterion = torch.nn.MSELoss()
    results=[]
    n_epoch = 200
    best_R = 0.0
    column=['epoch',
            'train_loss','valid_loss','test_loss',
            'train_c_index','valid_c_index','test_c_index',
            'train_RMSE','valid_RMSE','test_RMSE',
            'train_MAE','valid_MAE','test_MAE',
            'train_CORR','valid_CORR','test_CORR',
            'train_SD','valid_SD','test_SD'] #列表头名称
    
    epoch=0
    file_name = time_string + ".csv"
    for epoch in range(n_epoch):
        ll,evaluation1  = train(model, train_loader_compound, criterion, optimizer,epoch,device)
        if epoch%1==0:
            l,evaluation2 = valid(model, valid_loader_compound, criterion,device)
            l_, evaluation_ = valid(model, test_loader_compound, criterion,device)
            train_p=evaluation1['CORR']
            val_p=evaluation2['CORR']
            test_p=evaluation_['CORR']
            print(f'epoch {epoch+1} train_loss {ll:.5f} valid_loss {l:.5f} test_loss {l_:.5f}   train_CORR {train_p:.5f} valid_CORR {val_p:.5f} test_CORR {test_p:.5f}')
            result=[epoch+1,
               round(ll, 5),round(l, 5),round(l_, 5),#loss
               round(evaluation1['c_index'], 5),round(evaluation2['c_index'], 5),round(evaluation_['c_index'], 5),#c_index
               round(evaluation1['RMSE'], 5),round(evaluation2['RMSE'], 5),round(evaluation_['RMSE'], 5),#RMSE
               round(evaluation1['MAE'], 5),round(evaluation2['MAE'], 5),round(evaluation_['MAE'], 5),#MAE
               round(evaluation1['CORR'], 5),round(evaluation2['CORR'], 5),round(evaluation_['CORR'], 5),#CORR
               round(evaluation1['SD'], 5),round(evaluation2['SD'], 5),round(evaluation_['SD'], 5)]#SD 
            results.append(result)
            test=pd.DataFrame(columns=column,data=results)#将数据放进表格
            test.to_csv(file_name,mode='a' ) 
            clear_output()
            if evaluation_['CORR']>best_R:
                best_R= evaluation_['CORR']
                torch.save({'model': model.state_dict()}, 'model/model.pth')
            torch.save({'model': model.state_dict()}, 'model/model_finally.pth')
        scheduler.step()
        
    
    print("ok")
if __name__ == "__main__":
    main()