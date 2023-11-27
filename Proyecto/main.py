import clasificadores as cl

class MainUserInterface:

    def __init__(self) -> None:
        self.zoo = cl.Clasificador()
        self.zoo.read_Dataset()

    def Menu(self):
        print("------------------------Regresión logística------------------------")
        self.zoo.RL()
        print("------------------------K-Vecinos Cercanos------------------------")
        self.zoo.KNN()
        print("------------------------Maquinas Vector Soporte------------------------")
        self.zoo.SVM()
        print("------------------------Naive Bayes------------------------")
        self.zoo.NB()
        print("------------------------Red Neuronal------------------------")
        self.zoo.RN()

if __name__ == "__main__":
    menu = MainUserInterface()

    menu.Menu()