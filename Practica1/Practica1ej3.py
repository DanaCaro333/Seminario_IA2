import numpy as np

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.weights = []
        self.biases = []
        for i in range(1, len(layers)):
            self.weights.append(np.random.randn(layers[i], layers[i-1]))
            self.biases.append(np.random.randn(layers[i], 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def feedforward(self, inputs):
        activations = inputs
        for i in range(len(self.layers) - 1):
            activations = self.sigmoid(np.dot(self.weights[i], activations) + self.biases[i])
        return activations

    def train(self, inputs, targets, learning_rate, epochs):
        for epoch in range(epochs):
            for i in range(len(inputs)):
                
                # Feedforward
                activations = inputs[i]
                activation_list = [activations]
                for j in range(self.layers):
                    activations = self.sigmoid(np.dot(self.weights[j], activations) + self.biases[j])
                    activation_list.append(activations)

                # Backpropagation
                error = targets[i] - activations
                deltas = [error * self.sigmoid_derivative(activation_list[-1])]
                for j in range(len(self.layers) - 2, 0, -1):
                    delta = np.dot(self.weights[j].T, deltas[-1]) * self.sigmoid_derivative(activation_list[j])
                    deltas.append(delta)

                # Update weights and biases
                deltas.reverse()
                for j in range(len(self.layers) - 1):
                    self.weights[j] += learning_rate * np.dot(deltas[j], activation_list[j].T)
                    self.biases[j] += learning_rate * deltas[j]

# Ejemplo de uso
if __name__ == "__main__":
    # Definir la estructura de la red neuronal (por ejemplo, 2 capas ocultas con 4 neuronas cada una)
    layers = [2, 2, 1]  # 2 neuronas en la capa de entrada, 4 en las capas ocultas y 1 en la capa de salida

    # Crear una instancia de la red neuronal
    neural_network = NeuralNetwork(layers)

    # Datos de entrada y salida para el entrenamiento
    X1 = np.array([])
    X2 = np.array([])
    targets = np.array([])
    inputs = np.array([[],[]])
    
    with open("./info/concentlite.csv", "r") as arc:
        for line in arc.readlines():
            data = line.split(",")
            X1 = np.append(X1,float(data[0]))
            X2 = np.append(X2,float(data[1]))
            if int(data[2]) < 0:
                targets = np.append(targets,0)
            else:
                targets = np.append(targets,int(data[2]))
    inputs = np.array([X1,X2])  

    # Entrenamiento de la red neuronal
    learning_rate = 0.1
    epochs = 10000
    neural_network.train(inputs, targets, learning_rate, epochs)

    # Prueba de la red neuronal entrenada
    for i in range(len(inputs)):
        prediction = neural_network.feedforward(inputs[i])
        print(f"Input: {inputs[i]}, Prediction: {prediction}")