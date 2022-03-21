from helpers import *
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from create_dataset_fv import Dataset
import os
import argparse

#from preprocess_data import *

class Network(nn.Module):
    def __init__(self, tr_feat, tr_lbl, tr_sem, te_feat, te_lbl, te_sem, devise_name, marg=.1):
        super(Network, self).__init__()

        vis_dim = tr_feat.size(1)
        sem_dim = tr_sem.size(1)
        

        self.v = nn.Linear(vis_dim, sem_dim, bias=True).cuda()
        self.v.apply(init_weights)

        self.v1 = nn.Linear(sem_dim, sem_dim, bias=True).cuda()
        self.v1.apply(init_weights)

        self.v2 = nn.Linear(sem_dim, sem_dim, bias=True).cuda()
        self.v2.apply(init_weights)
        
        #self.v3 = nn.Linear(sem_dim, sem_dim, bias=True).cuda()
        #self.v3.apply(init_weights)

        self.s = nn.Linear(sem_dim, sem_dim, bias=True).cuda()
        self.s.apply(init_weights)

        self.s1 = nn.Linear(sem_dim, sem_dim, bias=True).cuda()
        self.s1.apply(init_weights)

        self.s2 = nn.Linear(sem_dim, sem_dim, bias=True).cuda()
        self.s2.apply(init_weights)

        #self.s3 = nn.Linear(sem_dim, sem_dim, bias=True).cuda()
        #self.s3.apply(init_weights)
        
        self.rrelu = torch.nn.RReLU()
        #self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
        self.drop = torch.nn.Dropout(p=0.2)
            

    def forward(self, x, s, devise_name):

        
        out_v = self.v(x.float())
        out_v = F.normalize(out_v, dim=1)
        out_v = self.rrelu(out_v)
        
        out_v = self.v1(out_v.float())
        out_v = F.normalize(out_v, dim=1)
        out_v = self.rrelu(out_v)
        
        #out_v = self.v2(out_v.float())
        #out_v = F.normalize(out_v, dim=1)
        #out_v = self.rrelu(out_v)


        #out_v = self.v3(out_v.float())
        #out_v = F.normalize(out_v, dim=1)
        #out_v = self.rrelu(out_v)

        out_s = self.s(s.float())
        out_s = F.normalize(out_s, dim=1)
        out_s = self.rrelu(out_s)

        out_s = self.s1(out_s.float())
        out_s = F.normalize(out_s, dim=1)
        out_s = self.rrelu(out_s)

        #out_s = self.s2(out_s.float())
        #out_s = F.normalize(out_s, dim=1)
        #out_s = torch.rrelu(out_s)

        #out_s = self.s3(out_s.float())
        #out_s = F.normalize(out_s, dim=1)
        #out_s = torch.rrelu(out_s)

        out = out_v.mm(out_s.t().float())
        return out

def init_weights(m):
    if isinstance(m, nn.Linear):
    	torch.nn.init.xavier_uniform_(m.weight)

def sample(x, y, bs):
    idx = np.random.choice(x.size(0), bs)
    return x[idx].cuda(), y[idx].cuda()

def validate(x, s, devise_name, model):
    model.eval()
    with torch.no_grad():
        out = model(x,s, devise_name)
        
    return out

def train_model(tr_feat, tr_lbl, tr_sem, te_feat, te_lbl, te_sem, devise_name, dir_path, args, bs=2000, nepoch=5, marg=.1):

    model=Network(tr_feat, tr_lbl, tr_sem, te_feat, te_lbl, te_sem, devise_name, marg=marg)
    opt   = torch.optim.Adam(model.parameters(), lr=args.lr)
    s_tr = tr_sem.cuda()
    
    model.train()
    t_ds = Dataset(os.path.join(args.data_dir,'1k'))
    tr_accs,tr_losses  = [],[]
   
    t_dl = DataLoader(t_ds, batch_size=bs, shuffle=False, sampler=RandomSampler(t_ds), num_workers=8)

    for ep in tqdm(range(nepoch), position=0, leave=True):
        for _, batch in enumerate(tqdm(t_dl, position=1, leave=True)):
            x = batch[0]
            y = batch[1]
            x = x.cuda()
            y = y.cuda()
            idx = torch.arange(0, y.size(0), dtype=torch.long, device="cuda")
            out = model(x, s_tr, devise_name)
            topk_v = torch.tensor(topk(out, y))
            val = out[idx, y].unsqueeze(1)
            zeros = torch.zeros_like(val)
            out = torch.max(zeros, marg-val+out)

            loss = out.mean()
            tr_losses.append(loss.item())
            tr_accs.append(topk_v[0].item())

            opt.zero_grad()
            loss.backward()
            opt.step()
        
        if 'b_psi' in args.sem_rep:
            dirname = dir_path + '_eps_' + str(args.eps) + '_tau_' + str(args.tau) + '_' + args.type + '_fv'
        else:
            dirname = dir_path + '_' + args.type + '_fv'

        if not os.path.exists(dirname):
            os.makedirs(dirname)

        if (ep+1) % 10 == 0 or ep==0:
            torch.save({'epoch': str(ep+1), 'state_dict': model.state_dict(), 'optimizer': opt.state_dict()}, os.path.join(dirname, 'devise_marg_' + str(marg) + '_lr_'+ str(args.lr) + '_epoch_' + str(ep+1)+'.pt'))
 
    tr_loss = np.mean(tr_losses)
    tr_acc = np.mean(tr_accs)

    return model, opt, tr_loss, tr_acc

def parse_args():
    parser = argparse.ArgumentParser(description="DeVise")
   
    # Arguments we added
    parser.add_argument('--output', default='devise_result', help='directory for result')
    parser.add_argument('--sem_rep', default='fine_tune_b_psi_eps_0.9_tau_0.96_pre_trained_fv/all_data_semantic_rep_after_train_epochs_5.pt', help='semantic representations')
    parser.add_argument('--sem_type', default='bert_p_w', help='type of semantic representations')
    parser.add_argument('--num_epochs', type=int, default=200, help='')
    parser.add_argument('--type', type=str, default='pre_trained', help='pretrain or finetuned')
    parser.add_argument('--lr', type=float, default=0.0004, help='learning_rate')
    parser.add_argument('--marg', type=float, default=0.2, help='devise marg')
    parser.add_argument('--bs', type=int, default=768, help='batch_size')
    parser.add_argument('--split', type=str, default='all_data', help='1k/2hop/3hop/all_data')
    parser.add_argument('--tau', type=float, default=0.96, help='tau')
    parser.add_argument('--eps', type=float, default=0.95, help='eps')
    parser.add_argument('--data_dir', default='', help='dataset directory')
    parser.add_argument('--d_name', default='imagenet', help='name of dataset')
    
    args = parser.parse_args()
    return args


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

args =  parse_args()

hie=args.split

############################## Training ##############################
print("Training...")

# 1K visual features / labels
x_tr=torch.load(os.path.join(args.data_dir,'1k_x.pt'))
y_tr=torch.load(os.path.join(args.data_dir,'1k_y.pt'))

# 1K semantic representations
if 'w2v' in args.sem_type:
    s_tr=torch.load(os.path.join(args.data_dir,'1k_sem.pt'))
elif 'w2v_10' in args.sem_type:
    s_tr=torch.load(os.path.join(args.data_dir,'1k_sem_10epochs.pt'))
elif 'exem' in args.sem_type:
    s_tr=torch.load(os.path.join(args.data_dir,'1k_sem_exem.pt'))
elif 'bert' in args.sem_type:
    if len(torch.load(args.sem_rep)) != 993:
        s_tr = torch.load(args.sem_rep)[0:993]
    else:
        s_tr = torch.load(args.sem_rep)

assert len(s_tr) == 993


model,opt, tr_loss, tr_acc= train_model(x_tr, y_tr, s_tr, None,None,None, None, dir_path=args.output, args=args, bs=args.bs, nepoch=args.num_epochs, marg=args.marg)

############################## Testing ##############################
print("testing...")
if args.d_name=='imagenet':
    if 'w2v' in args.sem_type:
        if args.split=='all_data':
            s_te=torch.load(os.path.join(args.data_dir,'our_all_data_sem.pt'))
    elif 'w2v_10' in args.sem_type:
        if args.split=='all_data':
            s_te=torch.load(os.path.join(args.data_dir,'our_all_data_sem_10epochs.pt'))
    elif 'exem' in args.sem_type:
        if args.split=='all_data':
            s_te=torch.load(os.path.join(args.data_dir,'our_all_data_sem_exem.pt'))
    elif 'bert' in args.sem_type:
        if args.split=='all_data':
            if len(torch.load(args.sem_rep)) != 14840:
                s_te =torch.load(args.sem_rep)[993:]
            else:
                s_te =torch.load(args.sem_rep)
                assert len(s_te) == 14840
        elif args.split=='3hop':
            if len(torch.load(args.sem_rep)) != 5984:
                s_te =torch.load(args.sem_rep)[993:6977]
            else:
                s_te =torch.load(args.sem_rep)
            assert len(s_te) == 5984

    test_accs = []
    acc_met = 'per_class'
    
    if args.split=='3hop':
        first=True
        bs=180
        num_class = len(s_te)
        per_class_acc = [None] * num_class
        for i in tqdm(range(num_class)):
            print(os.path.join(args.data_dir,'our_3hop_dir/' + str(i) + '/'))
            ds = Dataset(os.path.join(args.data_dir,'our_3hop_dir/' + str(i) + '/'))

            dl = DataLoader(ds, batch_size=bs, shuffle=False, sampler=SequentialSampler(ds), num_workers=8)
            with torch.no_grad():
                for j, batch in enumerate(dl):
                    x = batch[0]
                    y = batch[1]
                    out = validate(x.cuda(), s_te.cuda(), None, model)

                    tmp = torch.topk(out, 5)[1] == torch.unsqueeze(y.cuda(), dim=1)
                    tmp = tmp.float()
                    if acc_met =='per_samp':
                        if first:
                            per_samp_test_accs = tmp
                            first=False
                        else:
                            per_samp_test_accs = torch.cat((per_samp_test_accs, tmp), dim=0)
                      
                    elif acc_met =='per_class':
                        for y_ind in range(len(y)):
                            if per_class_acc[y[y_ind]] == None:
                                per_class_acc[y[y_ind]] = torch.unsqueeze(tmp[y_ind], dim=0)
                            else:
                                per_class_acc[y[y_ind]] = torch.cat((per_class_acc[y[y_ind]], torch.unsqueeze(tmp[y_ind], dim=0)), dim=0)
            
        if acc_met =='per_samp':
            final_per_samp_test_acc = torch.cumsum(per_samp_test_accs.mean(0)*100, 0).tolist()
            print("per_samp_accuracy: ", final_per_samp_test_acc)
        elif acc_met =='per_class':
            for acc_ind in range(len(per_class_acc)):
                per_class_acc[acc_ind] = per_class_acc[acc_ind].mean(0)

            per_class_acc = torch.stack(per_class_acc)
            
            final_per_class_test_acc = torch.cumsum(per_class_acc.mean(0)*100,0).tolist()
            print("3hop_per_class_top_1_5_accuracy: ", final_per_class_test_acc)

    elif args.split=='all_data':
        first=True
        bs=180
        num_class = len(s_te)
        print("num_class: ", num_class)
        per_class_acc = [None] * num_class
        for i in tqdm(range(num_class)):
            print(os.path.join(args.data_dir,'our_all_data_dir/' + str(i) + '/'))
            ds = Dataset(os.path.join(args.data_dir,'our_all_data_dir/' + str(i) + '/'))

            dl = DataLoader(ds, batch_size=bs, shuffle=False, sampler=SequentialSampler(ds), num_workers=8)
            with torch.no_grad():
                for j, batch in enumerate(dl):
                    x = batch[0]
                    y = batch[1]
                    out = validate(x.cuda(), s_te.cuda(), None, model)

                    tmp = torch.topk(out, 5)[1] == torch.unsqueeze(y.cuda(), dim=1)
                    tmp = tmp.float()
                    if acc_met =='per_samp':
                        if first:
                            per_samp_test_accs = tmp
                            first=False
                        else:
                            per_samp_test_accs = torch.cat((per_samp_test_accs, tmp), dim=0)
                       
                        
                    elif acc_met =='per_class':
                        for y_ind in range(len(y)):
                            if per_class_acc[y[y_ind]] == None:
                                per_class_acc[y[y_ind]] = torch.unsqueeze(tmp[y_ind], dim=0)
                            else:
                                per_class_acc[y[y_ind]] = torch.cat((per_class_acc[y[y_ind]], torch.unsqueeze(tmp[y_ind], dim=0)), dim=0)
                   
        if acc_met =='per_samp':
            final_per_samp_test_acc = torch.cumsum(per_samp_test_accs.mean(0)*100, 0).tolist()
            print("per_samp_accuracy: ", final_per_samp_test_acc)
        elif acc_met =='per_class':
            for acc_ind in range(len(per_class_acc)):
                per_class_acc[acc_ind] = per_class_acc[acc_ind].mean(0)
            per_class_acc = torch.stack(per_class_acc)          
            final_per_class_test_acc = torch.cumsum(per_class_acc.mean(0)*100,0).tolist()
            print("all_data_per_class_top_1_5_accuracy: ", final_per_class_test_acc)
