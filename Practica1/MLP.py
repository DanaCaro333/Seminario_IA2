from sklearn.neural_network import MLPClassifier
import numpy as np
import matplotlib.pyplot as plt
import perceptronMLP as p
from sklearn.model_selection import LeavePOut
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

class MClassifier:
        def __init__(self, num_inputs, hidden_layers, num_outputs, learning_rate, epochs):
            self.num_inputs = num_inputs
            self.hidden_layers = hidden_layers
            self.num_outputs = num_outputs
            self.learning_rate = learning_rate
            self.epochs = epochs
            
            
            self.Perceptron = p.MultilayerPerceptron(self.num_inputs, self.hidden_layers, self.num_outputs, self.learning_rate, self.epochs);


        def leav_one_out(self, pe, data, labels):
            pe = pe
            x, y = data, labels
    
            accuracies = []
            lpo = LeavePOut(pe)
    
            for train_indices, test_indices in lpo.split(x):
                X_train, X_test = x[train_indices], x[test_indices]
                y_train, y_test = y[train_indices], y[test_indices]
    
                # Inicializa y entrena el modelo en el conjunto de entrenamiento
                model = p.MultilayerPerceptron(self.num_inputs, self.hidden_layers, self.num_outputs, self.learning_rate, self.epochs);
    
                model.train(X_train, y_train)
                predictions = []
                for inputs in X_test:
                    prediction = model.predict(inputs)
                    predictions.append(prediction)
    
                # Calcula la precisión y almacénala
                accuracy = accuracy_score(y_test, predictions)
                accuracies.append(accuracy)
    
    
            average_accuracy = sum(accuracies) / len(accuracies)
            
            std_deviation = np.std(accuracies)
            
            error_esperado_porcentaje = (1 - average_accuracy) * 100
    
    
    
            if(pe > 1):
                print("Leave-K-Out")
            else:
                print("Leave-One-Out")
    
            print(f"Precisión en el conjunto de prueba: {average_accuracy * 100:.2f}%")
            
            print(f"Desviación Estándar: {std_deviation * 100:.2f}%")
    
            print(f"Error Esperado: {error_esperado_porcentaje}%")
        
        def Tipo(self, prediccion, real):
    
            if(prediccion[0] ==  real[0] and prediccion[1] ==  real[1] and prediccion[2] ==  real[2]):
    
                if(prediccion[0] == -1 and prediccion[1] == -1 and prediccion[2] == 1):
                    return "Setosa"
                elif(prediccion[0] == -1 and prediccion[1] == 1 and prediccion[2] == -1):
                    return "Versicolor"
                elif(prediccion[0] == 1 and prediccion[1] == -1 and prediccion[2] == -1):
                    return "Virginica"
            else:
                return "None"
            
        
        def Testeo(self, data, labels):
                
                correct_predictions = 0
                total_predictions = len(data)
        
                predicted_labels = []

                # Define las etiquetas para cada clase
                colors = ['red', 'green', 'blue']
                tipo = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
                
                for inputs, label in zip(data, labels):
                    prediction = self.Perceptron.predict(inputs)
                    predicted_labels.append(prediction)
                    
                    print(f"Entradas: {inputs}, Real: {label}, Predicción: {prediction}, Tipo: {self.Tipo(prediction, label)}")
                    
        
                    if self.Tipo(prediction, label) != "None":
                        plt.scatter(inputs[0], inputs[1], color=colors[np.argmax(label)])
                        correct_predictions += 1
        
                accuracy = correct_predictions / total_predictions
        
                print(f"Precisión en el conjunto de prueba: {accuracy * 100:.2f}%")
                
                self.leav_one_out(1,data,labels)
        
                self.leav_one_out(2,data,labels)
         
                for i, label in enumerate(tipo):
                    plt.scatter([], [], color=colors[i], label=label)
                    plt.legend()
                
                    plt.xlabel('Longitud del Sépalo (cm)')
                    plt.ylabel('Longitud del Pétalo (cm)')
                    plt.title('Distribución de clases en el dataset')
                    plt.show()
        def Entrenamiento(self, data, label):
        #Entrenamiento del perceptron
           self.Perceptron.train(data, label)
 
