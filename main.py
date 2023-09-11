import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold,train_test_split,StratifiedKFold
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
import seaborn
seaborn.set(style='whitegrid',font_scale=1.0)
from collections import Counter
from dataprepare import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--use-cuda', default=False,
                    help='CUDA training.')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Learning rate.')
parser.add_argument('--wd', type=float, default=1e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64,
                    help='Dimension of representations')
parser.add_argument('--data', type=int, default=1,
                    help='Dataset')               

args = parser.parse_args()
args.cuda = args.use_cuda and torch.cuda.is_available()

set_seed(args.seed,args.cuda)

path = './Data%d/' % args.data
df = pd.read_csv(path+'food-compound.csv')
f = pd.read_csv(path+'food.csv')
c = pd.read_csv(path+'compound.csv')
n = len(f)
m = len(c)
x = np.loadtxt(path+'x.txt')
if path == './Data1/': x = 0.1*(np.log10(x+1e-5)+5)
#s = normalized(np.loadtxt(path+'sim.txt')) # Tanimoto similarity
s = np.loadtxt(path+'fps.txt')
#x = x.dot(s)
y0 = f['category'].values
le = LabelEncoder()
y = le.fit_transform(y0)
weight = torch.zeros(max(y)+1)
for k,v in Counter(y).items():
    weight[k] = 1/np.log(v)

kf = StratifiedKFold(n_splits=2,shuffle=True)
cm = np.zeros((
    len(set(y)),len(set(y))
)).astype('int')

class GraphConv(nn.Module):
    # my implementation of GCN
    def __init__(self,in_dim,out_dim,drop=0.5,bias=False,activation=None):
        super(GraphConv,self).__init__()
        self.dropout = nn.Dropout(drop)
        self.activation = activation
        self.w = nn.Linear(in_dim,out_dim,bias=bias)
        nn.init.xavier_uniform_(self.w.weight)
        # self.bias = bias
        # if self.bias:
        #     self.b = nn.Parameter(torch.zeros(1, out_dim))
    
    def forward(self,adj,x):
        x = self.dropout(x)
        x = adj.mm(x)
        x = self.w(x)
        if self.activation:
            return self.activation(x)
        else:
            return x

class GNet(nn.Module):
    def __init__(self,in_dim,out_dim,hid_dim=args.hidden,bias=False):
        super(GNet,self).__init__()
        self.res1 = GraphConv(in_dim,hid_dim,bias=bias,activation=F.relu)
        self.res2 = GraphConv(hid_dim,out_dim,bias=bias,activation=None)
        self.lin = nn.Linear(s.shape[1],hid_dim)
    
    def forward(self,g,z,s):
        h = self.res1(g,z)
        s = self.lin(s)
        # attention
        h = torch.softmax(torch.mm(z,s)/np.sqrt(z.shape[1]),dim=-1)*h           
        return h,self.res2(g,h)

graph = norm_adj(x)
pred = np.zeros_like(y)
x = torch.from_numpy(x).float()
s = torch.from_numpy(s).float()
y = torch.from_numpy(y).long()

for train,test in kf.split(x,y):
    yt = y[train]
    yv = y[test]
    clf = GNet(x.shape[1],max(y)+1)
    if args.cuda:
        clf = clf.cuda()
        graph = graph.cuda()
        x = x.cuda()
        s = s.cuda()
        yt = yt.cuda()
        weight = weight.cuda()
    
    opt = torch.optim.Adam(clf.parameters(),lr=args.lr,weight_decay=args.wd)
    clf.train()
    for e in range(args.epochs):     
        h,z = clf(graph,x,s)
        loss = F.cross_entropy(z[train],yt,weight=weight)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if e%10 == 0 and e!=0:
            print('Epoch %d | Lossp: %.4f' % (e, loss.item()))
    
    clf.eval()
    h,z = clf(graph,x,s)
    if args.cuda:
        h = h.cpu().detach().numpy()
        z = z.cpu().detach().numpy()
    else:
        h = h.detach().numpy()
        z = z.detach().numpy()
    
    pred = z[test].argmax(axis=1)
    cm += confusion_matrix(yv,pred)
    print(accuracy_score(yv,pred))

lt = le.inverse_transform(range(max(y)+1))
plt.figure()
seaborn.heatmap(cm,annot=True,cbar=False,fmt='d',
    xticklabels=lt,
    yticklabels=lt)
plt.xlabel('Pred')
plt.ylabel('True')
plt.ylim(max(y)+1,0)
plt.tight_layout()
plt.show()

pca = PCA(n_components=2)
xx = pca.fit_transform(h)
seaborn.scatterplot(x=xx[:,0],y=xx[:,1],hue=y0)
plt.show()
a = np.corrcoef(h)
a[np.where(np.isnan(a))] = 0
np.savetxt(path+'foodsim.txt',a,fmt='%.3f')