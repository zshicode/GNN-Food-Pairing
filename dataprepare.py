import numpy as np
import pandas as pd
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

def prepareAll():
    prepare('./Data1/')
    prepare('./Data2/')
    prepare('./Data3/')

prepareAll()