import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from os import remove

"""
Created on Thu Sep 14 22:08:37 2023

@author: danacaro
"""
def particionar(dataset, e, pTrain, name):
    for i in range(e):
            
        df = pd.read_csv(dataset, header=None)
        p_train = pTrain / 100  
        df['is_train'] = np.random.uniform(0, 1, len(df)) <= p_train
        train, test = df[df['is_train']==True], df[df['is_train']==False]
        df = df.drop(['is_train'], axis=1)
        train.pop('is_train')
        test.pop('is_train')

        trainData = pd.DataFrame(train)
        testData = pd.DataFrame(test)
        
        trainData.to_csv("./info/"+name+str(i+1)+"_trn.csv",header=False, index=False)
        testData.to_csv("./info/"+name+str(i+1)+"_tst.csv",header=False, index=False)


def merge(e):
    for i in range(e):
        
        particionar("./info/spheres2d10.csv", 1, 80, "10mix_")
        particionar("./info/spheres2d50.csv", 1, 80, "50mix_")
        particionar("./info/spheres2d70.csv", 1, 80, "70mix_") 
        
        df10 = pd.DataFrame(pd.read_csv("./info/10mix_1_trn.csv", header=None))
        df50 = pd.DataFrame(pd.read_csv("./info/50mix_1_trn.csv", header=None))
        df70 = pd.DataFrame(pd.read_csv("./info/50mix_1_trn.csv", header=None))
        
        df10.merge(df50, how='inner')
        df10.merge(df70, how='inner')
        
        df10.to_csv("./info/Mix"+str(i+1)+"_trn.csv",header=False, index=False)
        
        df10 = pd.DataFrame(pd.read_csv("./info/10mix_1_tst.csv", header=None))
        df50 = pd.DataFrame(pd.read_csv("./info/50mix_1_tst.csv", header=None))
        df70 = pd.DataFrame(pd.read_csv("./info/50mix_1_tst.csv", header=None))
        
        df10.merge(df50, how='inner')
        df10.merge(df70, how='inner')
        
        remove("./info/10mix_1_trn.csv")
        remove("./info/50mix_1_trn.csv")
        remove("./info/70mix_1_trn.csv")
        
        remove("./info/10mix_1_tst.csv")
        remove("./info/50mix_1_tst.csv")
        remove("./info/70mix_1_tst.csv")
        
        df10.to_csv("./info/Mix"+str(i+1)+"_tst.csv",header=False, index=False)
        

merge(5)