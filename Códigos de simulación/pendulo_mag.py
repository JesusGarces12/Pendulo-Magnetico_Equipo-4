import numpy as np
import matplotlib.pyplot as plt
import math as mt
from mpl_toolkits.mplot3d import Axes3D

def runge_kutta_4th_order(f, y0, t0, tn, h):
    t_values = np.arange(t0, tn + h, h)
    y_values = [y0]

    for t in t_values[:-1]:
        k1 = h * np.array(f(t, y_values[-1]))
        k2 = h * np.array(f(t + h / 2, y_values[-1] + k1 / 2))
        k3 = h * np.array(f(t + h / 2, y_values[-1] + k2 / 2))
        k4 = h * np.array(f(t + h, y_values[-1] + k3))

        y_next = y_values[-1] + (k1 + 2*k2 + 2*k3 + k4) / 6
        y_values.append(y_next)

    return t_values, np.array(y_values)


def pendulo_magnetico(y, b, mu0, omega):
    x, y, z, vx, vy, vz = y

    ro = np.sqrt(x**2 + y**2)
    r = np.sqrt(ro**2 + z**2)

    fx = (1.45)**2*(mu0/(4*mt.pi))*((2*x/r**5)+(2*z**2-ro**2)*(-5*x/r**7))
    fy = (1.45)**2*(mu0/(4*mt.pi))*((2*y/r**5)+(2*z**2-ro**2)*(-5*y/r**7))
    fz = (1.45)**2*(mu0/(4*mt.pi))*((4*z/r**5)+(2*z**2-ro**2)*(-5*z/r**7))

    dxdt = vx
    dydt = vy
    dzdt = vz
    dvxdt = fx - b*vx-omega**2 * x
    dvydt = fy - b*vy-omega**2 * y
    dvzdt = fz - b*vz-omega**2 * z

    return [dxdt, dydt, dzdt, dvxdt, dvydt, dvzdt]

# Parámetros del sistema
omega = 5.51 

b=0.7

mu0=4*mt.pi*10**(-7)


# Condiciones iniciales
y0 = [0.2, -20.5, 5.0, 0.0, 0.0, 0.0]  # [x, y, z, vx, vy, vz]

# Tiempo de simulación
t0 = 0
tn = 15
h = 0.01

# Resuelve el sistema de ecuaciones diferenciales con Runge-Kutta de cuarto orden
t_values, y_values = runge_kutta_4th_order(
    lambda t, y: pendulo_magnetico(y, b, mu0, omega),
    y0,
    t0,
    tn,
    h
)

 # Grafica la trayectoria tridimensional del problema
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(t_values, y_values[:, 0], y_values[:, 1], label='Grafica sin nombre aun')
#ax.plot(y_values[:, 0], y_values[:, 1], y_values[:, 2])
ax.set_title('Trayectoria tridimensional del pendulo magnetico')
ax.set_xlabel('t')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.plot(0,0,0, marker="o", color="black")
plt.plot(y0[0],y0[1],y0[2], marker="o", color="red")
plt.show()

# print(y0[0],y0[1],y0[2])

# Graficar los resultados
plt.plot(y_values[:, 1], y_values[:, 4])
# plt.plot(t_values, y_values[:, 0], label='y')
# plt.plot(t_values, y_values[:, 2], label='z')
plt.xlabel('Posición y')
plt.ylabel('Velocidad y')
plt.title('Diagrama de fase')
plt.show()

# Graficar los resultados
plt.plot(t_values, y_values[:, 0], label='x')
plt.plot(t_values, y_values[:, 1], label='y')
plt.plot(t_values, y_values[:, 2], label='z')
plt.xlabel('Tiempo de oscilación')
plt.ylabel('Soluciones de la ecuación diferencial')
plt.title('Variación de las posiciones del péndulo en función del tiempo')
plt.legend()
plt.show()

# Graficar los resultados
plt.plot(t_values, y_values[:, 3], label='vx')
plt.plot(t_values, y_values[:, 4], label='vy')
plt.plot(t_values, y_values[:, 5], label='vz')
plt.xlabel('Tiempo de oscilación')
plt.ylabel('Soluciones de la ecuación diferencial')
plt.title('Variación de las velocidad del péndulo en función del tiempo')
plt.legend()
plt.show()

