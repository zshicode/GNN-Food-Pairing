import numpy as np
import pandas as pd
import torch
from rdkit import Chem,DataStructs
from rdkit.Chem import AllChem

fpLength = 1024

def ecfp(smiles):
    if smiles != 'Nutrient':
        mol = Chem.MolFromSmiles(smiles)
        if mol: 
            return AllChem.GetMorganFingerprintAsBitVect(mol,3,fpLength)
            #return Chem.RDKFingerprint(mol,fpSize=fpLength)
        else: 
            print(smiles)
    else:
        return 'Nutrient'

def prepare(path):
    df = pd.read_csv(path+'food-compound.csv')
    f = pd.read_csv(path+'food.csv')
    c = pd.read_csv(path+'compound.csv')
    n = len(f)
    m = len(c)
    x = np.zeros((n,m))
    for _,row in df.iterrows():
        if path == './Data2/':
            x[row['Food'],row['Compound']] = 1
        else:
            i = f.loc[f.name==row['Food']].index[0]
            j = c.loc[c.name==row['Compound']].index[0]
            if row['Mean'] != 'Not': x[i,j] = row['Mean']
            
    np.savetxt(path+'x.txt',x,fmt='%.3f')
    fp = c['smiles'].apply(ecfp)
    s = np.zeros((m,fpLength))
    for i in range(m):
        if fp[i] != 'Nutrient':
            arr = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(fp[i],arr)
            s[i] = arr
        else:
            s[i,i%6] = 1
    
    np.savetxt(path+'fps.txt',s,fmt='%.3f')

    # s = np.eye(m)
    # for i in range(m):
    #     for j in range(i):
    #         if fp[i] != 'Nutrient' and fp[j] != 'Nutrient':
    #             s[i,j] = DataStructs.FingerprintSimilarity(fp[i],fp[j])
    #             s[j,i] = s[i,j]
    
    # np.savetxt(path+'sim.txt',s,fmt='%.3f')

def set_seed(seed,cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

def neighborhood(feat,k,spec_ang=False):
    # compute C
    featprod = np.dot(feat.T,feat)
    smat = np.tile(np.diag(featprod),(feat.shape[1],1))
    if spec_ang: dmat = 1 - featprod/np.sqrt(smat*smat.T) # 1 - spectral angle
    else: dmat = smat + smat.T - 2*featprod
    dsort = np.argsort(dmat)[:,1:k+1]
    C = np.zeros((feat.shape[1],feat.shape[1]))
    for i in range(feat.shape[1]):
        for j in dsort[i]:
            C[i,j] = 1.0
    
    return C

def normalized(wmat):
    deg = np.diag(np.sum(wmat,axis=0))
    degpow = np.power(deg,-0.5)
    degpow[np.isinf(degpow)] = 0
    W = np.dot(np.dot(degpow,wmat),degpow)
    return W

def norm_adj(feat):
    C = neighborhood(feat.T,k=5,spec_ang=False)
    norm_adj = normalized(C.T*C+np.eye(C.shape[0]))
    g = torch.from_numpy(norm_adj).float()
    return g

if __name__ == '__main__':
    prepare('./Data1/')
    prepare('./Data2/')
    prepare('./Data3/')