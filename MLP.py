#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 22:44:46 2023

@author: danacaro
"""


import matplotlib.pyplot as plt
from numpy.random import randint
from numpy import linspace
import math

# Ejemplo del gradiente descendente aplicado a la función y = 10 - math.exp(-(x_1**2 + 3*x_2**2))
# La ecuación matemática para el gradiente (derivada) es = math.exp(-x_1-3*x_2)

x_inicial = randint(10) 
alpha = 0.1
n_iteraciones = 15

iteraciones = []
y = []

x_1 = x_inicial

for i in range(n_iteraciones):
	print('------------------------')
	print('iteración ', str(i+1))
    x_2 = x_1
    
	# Calcular gradiente
	gradiente = math.exp(-x_1-3*x_2)

	# Actualizar "x" usando gradiente descendente
	x_1 = x_1 - alpha*gradiente

	# Almacenar iteración y valor correspondiente
	y.append(10 - math.exp(-(x_1**2 + 3*x_2**2)))
	iteraciones.append(i+1)

	# Imprimir resultados
	print('x = ', str(x), ', y = ', str(10 - math.exp(-(x_1**2 + 3*x_2**2))))

plt.subplot(1,2,1)
plt.plot(iteraciones,y)
plt.xlabel('Iteración')
plt.ylabel('y')

X = linspace(-5,5,100)
Y = X**2 + 1
plt.subplot(1,2,2)
plt.plot(X,Y,0.0,1.0,'ro')
plt.xlabel('x')
plt.ylabel('y')

plt.show()