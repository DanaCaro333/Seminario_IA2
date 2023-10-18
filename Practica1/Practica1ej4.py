import MLP as mp
import perceptronMLP as p
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

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


particionar("./info/irisbin.csv",1, 80, "iris")

#Grafico Sepal - Longitud vs Ancho
iris = pd.read_csv("./info/iris1_trn.csv",header=None, names=['x1', 'x2', 'x3', 'x4', 'r1', 'r2', 'r3'])
print (iris)       

data = iris.drop(['r1','r2','r3'], axis=1)
data = preprocessing.normalize(data, axis=0)
#data = data.to_numpy()
label = iris.drop(['x1','x2','x3', 'x4'], axis=1)
label = label.to_numpy()

mlp = mp.MClassifier(num_inputs = 4, hidden_layers = [8], num_outputs = 3, learning_rate = 0.1, epochs = 100)

mlp.Entrenamiento(data, label)

irisT = pd.read_csv("./info/iris1_tst.csv",header=None, names=['x1', 'x2', 'x3', 'x4', 'r1', 'r2', 'r3'])
data = irisT.drop(['r1','r2','r3'], axis=1)
data = preprocessing.normalize(data, axis=0)
#data = data.to_numpy()
esperado = irisT.drop(['x1','x2','x3', 'x4'], axis=1)
esperado = esperado.to_numpy()

#prediccion = mlp.predict(data)
mlp.Testeo(data, label)
#print("Reporte del clasificador: \n %s\n %s\n"%(mlp, metrics.classification_report(esperado,prediccion)))
#print("Matriz de confusion:\n%s" % metrics.confusion_matrix(esperado,prediccion))


fig = iris[iris.r1 == 1].plot(kind='scatter',
          x='x4', y='x2', color='blue', label='Setosa')
iris[iris.r2 == 1].plot(kind='scatter',
   x='x4', y='x2', color='green', label='Versicolor', ax=fig)
iris[iris.r3 == 1].plot(kind='scatter',
    x='x4', y='x2', color='red', label='Virginica', ax=fig)

fig.set_xlabel('Sépalo - Longitud')
fig.set_ylabel('Sépalo - Ancho')
fig.set_title('Sépalo - Longitud vs Ancho')
plt.show()



