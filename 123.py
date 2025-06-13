import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import time
from numba import njit, prange
import os

# Набор значений Q


import argparse

# Чтение аргументов командной строки
parser = argparse.ArgumentParser(description='Solve the ODE system.')
parser.add_argument('--N', type=int, default=50, help='Number of particles')
parser.add_argument('--t_max', type=int, default=2500, help='Maximum simulation time')
parser.add_argument('--Q', type=int, default=5, help='Parameter Q')
parser.add_argument('--delta', type=float, default=0.05, help='Parameter delta')
parser.add_argument('--dt',type=float,default=0.01)
parser.add_argument('--gam',type=float,default=1.5)
parser.add_argument('--v_0',type=float,default=0.12)
parser.add_argument('--stepen',type=int,default=3)
args = parser.parse_args()


# Параметры

N = args.N
t_max = args.t_max
Q = args.Q
delta = args.delta
dt = args.dt
gam = args.gam
v_0=args.v_0
stepen = args.stepen

Q_values = [Q]
#N = 100
L = 1.0
f = 0.8
kappa = 5.2
alpha = 1.457
output_dir = "./"

# Создаем директорию для результатов, если её нет
os.makedirs(output_dir, exist_ok=True)

# Цикл по всем значениям Q
for Q in Q_values:
    start_time = time.perf_counter()
    print(f"Начало расчета для Q = {Q}")

    # Параметры, зависящие от Q
    gamma = 0.3 * Q ** (-3 / 2)
    V0 = 0.05 * Q ** (1 / 2)


    @njit(parallel=True, fastmath=False)
    def system(y, t):
        """Система ОДУ с параллельными вычислениями"""
        x = y[:N]
        v = y[N:2 * N]
        phi = y[2 * N:3 * N]

        # Уравнения движения
        dxdt = -v
        dvdt = gamma * (V0 ** 2 - v ** 2) * v + np.sin(x) - f

        # Параллельное вычисление взаимодействий
        dphidt = np.zeros(N)
        for n in prange(N):
            interaction = 0.0
            for m in range(N):
                dx = x[m] - x[n]
                dphi = phi[m] - phi[n] - alpha
                interaction += G(dx) * np.sin(dphi)
            dphidt[n] = interaction

        result = np.empty(3 * N)
        result[:N] = dxdt
        result[N:2 * N] = dvdt
        result[2 * N:] = dphidt
        return result


    @njit(fastmath=False)
    def G(x):
        """Периодическое ядро взаимодействия"""
        x_mod = (x + L / 2) % L - L / 2
        return kappa * np.cosh(kappa * (np.abs(x_mod) - L / 2)) / (2 * np.sinh(kappa * L / 2))


    # Генерация начальных условий
    np.random.seed(42)
    x0 = np.random.uniform(0, L, N)
    v0 = np.random.normal(0, 0.1, N)
    phi0 = np.random.uniform(-np.pi, np.pi, N)
    y0 = np.concatenate((x0, v0, phi0))

    # Интеграция системы
    t = np.arange(0, t_max + 1, dt)
    sol = odeint(system, y0, t, rtol=1e-6)

    # Сохранение результатов
    filename = os.path.join(output_dir, f'solution_N{N}_T{t_max}_Q{Q}.npy')
    np.save(filename, sol)

    print(f"Q={Q} завершен за {time.perf_counter() - start_time:.2f} сек. Файл: {filename}")