import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import time
from numba import njit, prange
import argparse
from datetime import datetime
start_time = time.perf_counter()  # Начало замера
'''
# Чтение аргументов командной строки
parser = argparse.ArgumentParser(description='Solve the ODE system.')
parser.add_argument('--N', type=int, default=512, help='Number of particles')
parser.add_argument('--t_max', type=int, default=5000, help='Maximum simulation time')
parser.add_argument('--Q', type=int, default=5, help='Parameter Q')
parser.add_argument('--dt',type=float,default=0.01)
parser.add_argument('--gam',type=float,default=1.5)
parser.add_argument('--v_0',type=float,default=0.12)
parser.add_argument('--stepen',type=int,default=3)
args = parser.parse_args()


# Параметры

N = args.N
t_max = args.t_max
Q = args.Q
dt = args.dt
gam = args.gam
v_0=args.v_0
stepen = args.stepen
'''
# Параметры, не зависящие от Q
N = 50
L = 1.0
f = 0.8
kappa = 5.2
alpha = 1.457
t_max = 5000
dt = 0.01
delta = 0.05
gam=0.3
v_0=0.3
Q = 1
output_dir = f"C:/work/"

gamma = gam * Q ** (-1 / 2)
V0 = v_0 * Q ** (1 / 2)
print(gamma)
print(V0)
@njit(fastmath=False)
def G(x):
    """Периодическое ядро взаимодействия"""
    x_mod = (x + L / 2) % L - L / 2
    return kappa * np.cosh(kappa * (np.abs(x_mod) - L / 2)) / (2 * np.sinh(kappa * L / 2))


c_cnt = 0

@njit(parallel=True, fastmath=False)
#@njit(parallel=True, fastmath=True)
def system(y, t):
#    global c_cnt
#    c_cnt = c_cnt + 1

#    if ((c_cnt % 1000) == 0):
#        print (c_cnt, time.time())

    """Система ОДУ с параллельными вычислениями"""
    x = y[:N]
    v = y[N:2 * N]
    phi = y[2 * N:3 * N]

    # Уравнения движения
    dxdt = -v
    dvdt = gamma * (V0 ** 2 - v ** 2) * v + np.sin(x) - f

    # Параллельное вычисление взаимодействий
    dphidt = np.zeros(N)
    for n in prange(N):  # Распараллеливание по частицам
        interaction = 0.0
        for m in range(N):
            dx = x[m] - x[n]
            dphi = phi[m] - phi[n] - alpha
            interaction += G(dx) * np.sin(dphi)
        dphidt[n] = interaction

    # Собираем результат
    result = np.empty(3 * N)
    result[:N] = dxdt
    result[N:2 * N] = dvdt
    result[2 * N:] = dphidt

    return result



np.random.seed(42)
x0 = np.random.uniform(0, L, N)
v0 = np.random.normal(0, 0.1, N)
phi0 = np.random.uniform(-np.pi, np.pi, N)
y0 = np.concatenate((x0, v0, phi0))
# Дальнейший код остается без изменений...

# Интеграция
t = np.arange(0, t_max+1 , dt)

ts = time.perf_counter()
sol = odeint(system, y0, t, rtol=1e-6)
print(len(sol[:,:N]))
te = time.perf_counter()  # Конец замера
ex = te - ts
print("интегрирование завершено за:", ex)
time_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
np.save(f'---solution_N{N}_T{t_max}_Q{Q}_gamma{gam}_V0{v_0}.npy', sol)
