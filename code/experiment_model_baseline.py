import sys; sys.path.insert(0,'/home/lgc/rpvoltsag/stage01')
import numpy as np, os
from electrical_model import build_network, DEFAULT_DATA, steering_dictionary
from experiment_localization import load_data, hop_index
BASE='/home/lgc/rpvoltsag/stage01'
net=build_network(os.path.join(BASE,'BUSTOPO.csv'),os.path.join(BASE,'LINETOPO.csv'),os.path.join(BASE,'TRANSTOPO.csv'),os.path.join(BASE,DEFAULT_DATA,'DT_LINE.csv'),os.path.join(BASE,DEFAULT_DATA,'DT_TRANS.csv'))
N=net['n']; D,fid,ft,Nn=load_data(net); hop,G=hop_index(net)
Sfull=steering_dictionary(net['Zbus'])
E=len(fid)
def eval_model(monitors):
    Sm=Sfull[monitors,:]                      # (M,N)
    Smn=Sm/(np.linalg.norm(Sm,axis=0,keepdims=True)+1e-12)
    Y=D[:,monitors]; Yn=Y/(np.linalg.norm(Y,axis=1,keepdims=True)+1e-12)
    sim=Yn@Smn                                # (E,N)
    pred=np.argmax(sim,axis=1)
    e=np.array([min(hop[int(pred[t])].get(int(fid[t]),999),999) for t in range(E)])
    return float(np.mean(e<=1)),float(np.mean(e<=3)),float(np.mean(e==0))
for M in [100,N]:
    monitors=np.random.RandomState(0).choice(N,M,replace=False) if M<N else np.arange(N)
    l1,l3,ex=eval_model(monitors)
    print(f'model-only M={M}: <=1hop={l1*100:.1f}% <=3hop={l3*100:.1f}% exact={ex*100:.1f}%')
