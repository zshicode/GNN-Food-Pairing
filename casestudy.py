import numpy as np
import pandas as pd
import re

path = './Data1/'
a = np.loadtxt(path+'foodsim.txt')
f = pd.read_csv(path+'food.csv')
def query(s,a,f):
    ff = f[f['name'].str.contains(s,flags=re.IGNORECASE)]
    for idx,row in ff.iterrows():
        score = pd.DataFrame(columns=['name','category','score'])
        score['name'] = f['name']
        score['category'] = f['category']
        score['score'] = a[idx]
        score = score.sort_values(by='score',ascending=False)
        score.to_csv('score-'+row['name']+'.csv',index=False)

query('Green tea',a,f)