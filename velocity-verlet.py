import numpy as np
import matplotlib.pyplot as plt

def a(r):
  D = 5
  ao = 1
  rl = 1
  return D*2*ao*(np.exp(-2*ao*(r-rl)) - np.exp(-ao*(r-rl)))
  
def U(r):
  D = 5
  ao = 1
  rl = 1
  return D*(np.exp(-2*ao*(r-rl)) - 2*np.exp(-ao*(r-rl)))

n = 1000
h = 0.01

x0 = 4 #para energia cerca del minimo
vx0 = 0.1
t0 = 0.0

#Inicialización de arreglos
t = np.zeros(n+1)
x = np.zeros(n+1)
vx = np.zeros(n+1)

t[0] = t0
x[0] = x0
vx[0] = vx0


for i in range(n):
  t[i+1] = (i+1)*h
  x[i+1]= x[i] + vx[i]*h + (h*h/2)*a(x[i])
  vx[i+1] = vx[i] + (h/2)*(a(x[i])+a(x[i+1]))
  
for i in range(n+1):
  print(f"{t[i]:.5f} {x[i]:.5f} {vx[i]:.5f}")
   

plt.figure(1)
plt.plot(t, x, 'o', markersize = 1)
plt.xlabel('t')
plt.ylabel('x')
plt.title('Posición vs Tiempo')
plt.figure(2)
plt.plot(t, vx,'o', markersize = 1)
plt.xlabel('t')
plt.ylabel('vx')
plt.title('Velocidad vs Tiempo')
plt.figure(3)
plt.plot(x,vx, 'o',markersize = 1)
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('x')
plt.ylabel('vx')
plt.title('Espacio fase')
plt.show()

# Crear un rango de valores para r
r_values = np.linspace(0, 5, 500)

# Calcular los valores de la función a(r)
a_values = U(r_values)

# Graficar la función
plt.plot(r_values, a_values)
plt.xlabel('r')
plt.ylabel('U(r)')
plt.title('Potencial de Morse U(r)')
plt.grid(False)
plt.show()