import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Adiciona o diretório pai ao sys.path para importar Functions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Functions.compute_cost import compute_cost
from Functions.gradient_descent import gradient_descent

# Ajuste o caminho para o arquivo de dados conforme necessário
data_path = os.path.join(os.path.dirname(__file__), '..', 'Data', 'ex1data1.txt')
data = np.loadtxt(data_path, delimiter=',')

x = data[:, 0]
y = data[:, 1]
m = y.size

# Adiciona coluna de 1s para o termo de bias
x_aug = np.stack([np.ones(m), x], axis=1)

# Parâmetros do experimento
alpha = 0.01
iterations = 1500

# Inicializações fixas 
np.random.seed(42)  # Para reprodutibilidade
initial_thetas = [
    np.array([8.5, 4.0]),
    np.array([5.0, 5.0]),
    np.array([-5.0, 5.0]),
]

labels = [
    " [8.5,4.0]",
    " [5,5]",
    " [-5,5]",
]

trajectories = []

for theta_init, label in zip(initial_thetas, labels):
    _, _, theta_history = gradient_descent(x_aug, y, theta_init, alpha, iterations)
    trajectories.append((theta_history, label))

# Gera a grade para o gráfico de contorno e superfície
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)
j_vals = np.zeros((theta0_vals.size, theta1_vals.size))

for i, t0 in enumerate(theta0_vals):
    for j, t1 in enumerate(theta1_vals):
        t = np.array([t0, t1])
        j_vals[i, j] = compute_cost(x_aug, y, t)

j_vals = j_vals.T  # Para o plot funcionar corretamente

# Gráfico de contorno 2D
plt.figure(figsize=(8, 6))
CS = plt.contour(theta0_vals, theta1_vals, j_vals, levels=np.logspace(-2, 3, 20), cmap='jet')
plt.clabel(CS, inline=1, fontsize=8)

for theta_history, label in trajectories:
    plt.plot(theta_history[:, 0], theta_history[:, 1], marker='o', markersize=3, label=label)

plt.xlabel(r'$\theta_0$')
plt.ylabel(r'$\theta_1$')
plt.title('Trajetórias do Gradiente Descendente para Diferentes Inicializações')
plt.legend()
plt.grid(True)
plt.xlim(-10, 10)
plt.ylim(-1, 4)
plt.tight_layout()
plt.show()

# Gráfico de superfície 3D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

t0_mesh, t1_mesh = np.meshgrid(theta0_vals, theta1_vals)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(t0_mesh, t1_mesh, j_vals, cmap='viridis', edgecolor='none', alpha=0.7)
ax.set_xlabel(r'$\theta_0$')
ax.set_ylabel(r'$\theta_1$')
ax.set_zlabel('Custo')
ax.set_title('Superfície da Função de Custo J($\\theta_0$, $\\theta_1$)')

# Trajetórias das inicializações no 3D
for theta_history, label in trajectories:
    # Calcula o custo ao longo da trajetória
    costs = np.array([compute_cost(x_aug, y, th) for th in theta_history])
    ax.plot(theta_history[:, 0], theta_history[:, 1], costs, marker='o', markersize=3, label=label)

ax.legend()
plt.tight_layout()
plt.show()