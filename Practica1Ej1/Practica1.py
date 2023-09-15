import numpy as np
import matplotlib.pyplot as plt

class perceptron:

    def __init__(self, n_input, learning_rate):
        self.w = -1 + 2*np.random.rand(n_input)
        self.b = -1 + 2*np.random.rand()
        self.eta = learning_rate

    def predict(self, X):
        p = X.shape[1]
        y_est = np.zeros(p)
        for i in range(p):
            y_est[i] = np.dot(self.w, X[:, i]) + self.b
            if y_est[i] >= 0:
                y_est[i] = 1
            else:
                y_est[i] = 0
        return y_est
    
    def fit(self, X, Y, epochs=20):
        p = X.shape[1]
        for _ in range(epochs):
            for i in range(p):
                y_est = self.predict(X[:,i].reshape(-1,1))
                self.w += self.eta * (Y[i] - y_est)*X[:,i]
                self.b += self.eta * (Y[i] - y_est)
                

def draw_perceptron(model):
    w1,w2,b = model.w[0], model.w[1], model.b
    plt.plot([-2,2], [(1/w2)*(-w1*(-2)-b), (1/w2)*(-w1*(2)-b)])

X1 = np.array([])
X2 = np.array([])
Y = np.array([])
X = np.array([[],[]])

with open("./info/XOR_trn.csv", "r") as arc:
    for line in arc.readlines():
        data = line.split(",")
        X1 = np.append(X1,float(data[0]))
        X2 = np.append(X2,float(data[1]))
        if int(data[2]) < 0:
            Y = np.append(Y,0)
        else:
            Y = np.append(Y,int(data[2]))
X = np.array([X1,X2])   

model = perceptron(2, 0.5)
model.fit(X, Y)

X1 = np.array([])
X2 = np.array([])
Y = np.array([])
X = np.array([[],[]])

with open("./info/XOR_tst.csv", "r") as arc:
    for line in arc.readlines():
        data = line.split(",")
        X1 = np.append(X1,float(data[0]))
        X2 = np.append(X2,float(data[1]))
        if int(data[2]) < 0:
            Y = np.append(Y,0)
        else:
            Y = np.append(Y,int(data[2]))
X = np.array([X1,X2])   

print(model.predict(X))

#Dibujo
p = X.shape[1]
for i in range(p):
    if Y[i] == 0:
        plt.plot(X[0, i], X[1, i], 'or')
    else:
        plt.plot(X[0, i], X[1, i], 'ob')

plt.title('Perceptron')
plt.grid('on')
plt.xlim([-2,2])
plt.ylim([-2,2])
plt.xlabel(r'$x_1&')
plt.ylabel(r'$x_2$')
draw_perceptron(model)
plt.show()

