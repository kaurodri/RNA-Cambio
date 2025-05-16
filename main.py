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

#Célula 8 - Função de perda (MSE)
def mse(Y, y):
    return (Y - y) ** 2

#Célula 9 - Backpropagation
def backpropagation(inputs, outputs, targets, w, b, lr):
    wdx = lr * (-2 * inputs * (targets - outputs)).mean()
    bdx = lr * (-2 * (targets - outputs)).mean()
 
    w -= wdx
    b -= bdx
    return w, b

#Célula 10 - Treinamento do modelo
def model_fit(inputs, target, w, b, epochs=200, lr=0.001):
    for epoch in range(epochs):
        outputs = forward(inputs, w, b)
        loss = np.mean(mse(target, outputs))
        w, b = backpropagation(inputs, outputs, target, w, b, lr)

        if (epoch + 1) % (epochs / 10) == 0:
            print(f'Epoch: [{(epoch+1)}/{epochs}] Loss: [{loss:.6f}]')
    return w, b

#Célula 11
x = np.arange(1, 20, 2)
Y = get_linear_curve(x, 0.18, 0, ruido=0.2)

#Célula 12 - Inicializar pesos aleatórios
w = np.random.rand(1)
b = np.zeros(1)

#Célula 13 - Treinar o modelo
w, b = model_fit(x, Y, w, b, epochs=50, lr=0.001)
print(f'w (Taxa de câmbio aprendida): {w[0]:.5f}')
print(f'b (Viés aprendido): {b[0]:.5f}')
