import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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

#particionar("./info/spheres1d10.csv", 5, 80, "Ej2P1_")

particionar("./info/spheres1d10.csv", 5, 80, "Ej2P1_")

