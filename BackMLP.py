import numpy as np
import pandas as pd
from numpy import random
import matplotlib.pyplot as plt

class MLP():
    def _init_(self, zi, d, w_1, w_2, us, uoc, precision, epocas, fac_ap, n_ocultas, n_entradas, n_salidas):
        
def tanh(x):
    return np.than(x)

def dtanh(x):
    return 1.0 - np.tanh(x)**2

def sigmoide(x):
    return 1/(1+np.exp(-x))

def dsigmoide(x):
    s = dtahn(x)
    return s * (1-s)

def Datos_entrenamiento(matriz, x1, xn):
    xin = matriz[:,x1:xn+1]
    return xin

def Datos_validacion(matriz, xji, xjn):
    xjn = matriz[:,xji:xjn+1]
    return xjn

if "__main__" == __name__:
      |