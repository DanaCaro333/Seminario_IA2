import Clasificadores as cl

def showResults():
        pima = cl.Clasificadores(1)
        wine = cl.Clasificadores(2)

        pima.read_Dataset()
        print("RESULTADOS DE DATAFRAME DIABETES")
        pima.RL() 
        pima.KNN() 
        pima.SVM()     
        pima.NB()  
        pima.redN()
        
        wine.read_Dataset()
        print("RESULTADOS DE DATAFRAME CALIDAD DE VINO")
        wine.RL()
        wine.KNN()
        wine.SVM()
        wine.NB()
        wine.redN()

showResults()
