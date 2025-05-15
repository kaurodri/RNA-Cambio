# RNA - Python
import sys
sys.executable

#Célula 1
import numpy as np
import matplotlib.pyplot as plt

#Célula 2
plt.style.use('dark_background')
plt.rcParams['figure.figsize'] = (10, 8)

#Célula 3
def get_linear_curve(x, w, b=0, ruido=0):
    return w * x + b + ruido * np.random.randn(x.shape[0])

#Célula 4 - Dataset
x = np.arange(1, 101, 3)
Y = get_linear_curve(x, 0.18, 0, ruido=0.2)
#x.shape
#Y.shape
#O ruído introduz variações pequenas e aleatórias, tornando os dados mais parecidos com os de um dataset real.

#Célula 5
plt.scatter(x, Y)
plt.xlabel('BRL (Reais)', fontsize=20)
plt.ylabel('USD (Dólares)', fontsize=20)
plt.title('Conversão BRL → USD (Taxa ~0.20)')
plt.show()

#Célula 6
w = np.random.rand(1)
b = 0

#Célula 7 - Forward pass (predição)
def forward(inputs, w, b):
    return w * inputs + b
