import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier

class Clasificador:
    def __init__(self) -> None:
        self.dataset = None
        self.animal_class = {}

    def RL(self):#Regresión logistica
        x = self.dataset.drop(['Animal', 'Class'], axis=1)
        y = self.dataset['Class']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        model = LogisticRegression()
        model.fit(x_train, y_train)
        # Incio de predicciones
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        self.print_result(report, y_pred, accuracy)




    def KNN(self):#K vecinos

        x = self.dataset.drop(['Animal', 'Class'], axis=1)
        y = self.dataset['Class']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        k = 3
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        self.print_result(report, y_pred, accuracy)


    def SVM(self):#Maquinas vector soporte

        x = self.dataset.drop(['Animal', 'Class'], axis=1)
        y = self.dataset['Class']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        model = SVC(kernel='sigmoid', C=1.0)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        self.print_result(report, y_pred, accuracy)

    def NB(self):#Naive Bayes
        x = self.dataset.drop(['Animal', 'Class'], axis=1)
        y = self.dataset['Class']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        model = MultinomialNB()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        self.print_result(report, y_pred, accuracy)

    def RN(self):#Red Neuronal
        x = self.dataset.drop(['Animal', 'Class'], axis=1)
        y = self.dataset['Class']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        model = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=1000)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        self.print_result(report, y_pred, accuracy)

    def print_result(self, report, y_pred, accuracy):
        print("Exactitud del modelo: {:.2f}".format(accuracy))
        print("Informe de clasificación:")
        print(report)
        print(y_pred)

    def read_Dataset(self):

        self.dataset = pd.read_csv("zoo.data", header=None, names=[
            'Animal', 'Hair', 'Feathers', 'Eggs', 'Milk', 'Airborne', 'Aquatic', 'Predator',
            'Toothed', 'Backbone', 'Breathes', 'Venomous', 'Fins', 'Legs', 'Tail', 'Domestic', 'Catsize', 'Class'
        ])
