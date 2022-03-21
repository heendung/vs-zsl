# Data
import pickle
import h5py
import torch
import tqdm
import pandas as pd
import torch.nn.functional as F
from os.path import join as pj
from scipy.io import loadmat
import numpy as np

train_split = "../data/splits/train"
v1_split    = "../data/splits/goz_v1"
sem_dir     = "../data/semantics/"
train_feature_path = "../data/visuals/features/train.h5"
test_feature_path =  "../data/visuals/features/test.h5"

feature_path = "../data/visuals/features/"


def load_visuals(nodes, path, nsamp=None):
    """
        Reads Visual features from disk and return them as torch.Tensor
        Input:
            nodes (str): wnid identifiers of the visual classes to return
            path  (str): Path to the h5py store 
    """
    feats, lbls = [],[]
    with h5py.File(path, "r") as f:
        for i,wnid in tqdm.tqdm(enumerate(nodes)):
            feat  = f["images"][wnid][:nsamp]
            feats += [torch.from_numpy(feat)]
            lbls  += [i]*feat.shape[0]
            
    lbls = torch.LongTensor(lbls)
    feats= torch.cat(feats)
    return feats, lbls

def load_semantics(nodes, semantics, filter_out=True, normalize=False):
    """
    """
    df = pd.read_pickle(pj(sem_dir, semantics))
    if filter_out:
        nodes = list(filter(lambda x: x in df.index, nodes))
    x  = df.loc[list(nodes)].fillna(0).values
    x = torch.from_numpy(x).float()
    return F.normalize(x) if normalize else x, nodes

def load_train_set(semantics="glove", nsamp=500, norm_sem=False, filter_out=True, dataset_name="goz"):
    """
    """
    if dataset_name=="goz":
        trnodes = pickle.load(open(train_split, "rb"))
        sem, trnodes = load_semantics(trnodes, semantics, normalize=norm_sem, filter_out=filter_out)
        x, y = load_visuals(trnodes, train_feature_path, nsamp=nsamp)
        #if norm_sem:
        #    x = F.normalize(x)
    else:
        x, y = [],[]
        dataset = loadmat(feature_path + dataset_name)
        all_x = dataset['X']
        all_y = dataset['Y']
        for loc in dataset['tr_loc']:
            loc = loc[0]
            tmp = np.asarray([[all_x[loc-1]][0]])
            tmp = torch.from_numpy(tmp)
            x += [tmp]
            y.append(all_y[loc-1][0])
        y = torch.LongTensor(y)
        k = np.unique(y)
        ny = y.numpy()
        y_ = []
        for ele in ny:                                 
            y_.append(np.where(k==ele)[0][0])                        
        y = torch.LongTensor(y_)     
        x = torch.cat(x)
        #x = x.type(torch.float32)
        sem = []
        all_sem = dataset['attr2']
        for ele in k:
            tmp = np.asarray([[all_sem[ele-1]][0]])
            tmp = torch.from_numpy(tmp)
            sem += [tmp]
        sem = torch.cat(sem)
        #sem = sem.type(torch.float32)
        if norm_sem:                              
            #x = F.normalize(x)
            sem = F.normalize(sem)
    return x, y, sem

def load_test_set(semantics="glove", generalized=False, norm_sem=False, filter_out=True, dataset_name="goz"):
    """
    """
    if dataset_name=="goz":
        tenodes = pickle.load(open(v1_split, "rb"))
        if generalized:
            tenodes += pickle.load(open(train_split, "rb"))
        sem, tenodes = load_semantics(tenodes, semantics, normalize=norm_sem, filter_out=filter_out)
        x, y = load_visuals(tenodes, test_feature_path)
        #if norm_sem:                              
        #    x = F.normalize(x) 
    else:
        x, y = [],[]
        sem = []
        dataset = loadmat(feature_path + dataset_name)
        all_x = dataset['X']
        all_y = dataset['Y']
        for loc in dataset['te_loc']:
            loc = loc[0]
            tmp = np.asarray([[all_x[loc-1]][0]])
            tmp = torch.from_numpy(tmp)
            x += [tmp]
            y.append(all_y[loc-1][0])
        
        y = torch.LongTensor(y)
        k = np.unique(y)
        ny = y.numpy()
        y_ = []
        for ele in ny:
            y_.append(np.where(k==ele)[0][0])
        y = torch.LongTensor(y_) 
        x = torch.cat(x)
        #x = x.type(torch.float32)
        sem = []
        all_sem = dataset['attr2']
        for ele in k:
            tmp = np.asarray([[all_sem[ele-1]][0]])
            tmp = torch.from_numpy(tmp)
            sem += [tmp]
        sem = torch.cat(sem)
        #sem = sem.type(torch.float32)
        if norm_sem:                              
            #x = F.normalize(x)
            sem = F.normalize(sem)
    return x, y, sem

def h_score(te, tr):
    """
    """
    return 2/(1/te + 1/tr)

def h_scores(te, tr):
    """
    """
    return list(map(lambda x:h_score(x[0],x[1]), zip(te,tr)))

def topk(scores, lbl, k=1):
    """
    """
    return torch.cumsum((scores.topk(k)[1]==lbl.unsqueeze(1)).float().mean(0)*100, 0).tolist()

def pp(x):
    canvas =  "|".join([" {"+ str(i)+":.2f} " for i,y in enumerate(x)])
    print(canvas.format(*x))

