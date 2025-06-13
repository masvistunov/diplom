Великий магистр, [16.04.2025 22:05]
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import time
from numba import njit, prange
import argparse
import csv
import json
from pathlib import Path
from tqdm import tqdm



parser = argparse.ArgumentParser(description='Particle system simulation and visualization')
parser.add_argument('--input', type=str, help='Input CSV file with simulation data')
parser.add_argument('--output', type=str, help='Output CSV file to save results')
parser.add_argument('--params', type=str, help='JSON file for parameters')
parser.add_argument('--plot-type', choices=['snapshot', 'order', 'all'], default='all',
                    help='Type of plot to generate (default: all)')
args, _ = parser.parse_known_args()

parser.add_argument('--N', type=int, required=True)
parser.add_argument('--L', type=float, required=True)
parser.add_argument('--gamma', type=float, required=True)
parser.add_argument('--V0', type=float, required=True)
parser.add_argument('--f', type=float, required=True)
parser.add_argument('--kappa', type=float, required=True)
parser.add_argument('--alpha', type=float, required=True)
parser.add_argument('--t_max', type=float, required=True)
parser.add_argument('--dt', type=float, required=True)
parser.add_argument('--Q', type=int, required=True)
parser.add_argument('--delta', type=int, required=True)

args = parser.parse_args()
params = vars(args)
globals().update(params)


start_time = time.perf_counter()  # Начало замера

# Параметры
#N = 50
#L = 1.0
#gamma = 0.3
#V0 = 0.3
#f = 0.8
#kappa = 5.2
#alpha = 1.457
#t_max = 1000
#dt = 0.01
#Q = 25
#delta = 15




'''
# Векторизованная функция взаимодействия
@njit
def G(x):
    x_mod = (x + L/2) % L - L/2
    return kappa * np.cosh(kappa * (np.abs(x_mod) - L/2)) / (2 * np.sinh(kappa * L/2))

# Оптимизированная система уравнений
@njit
def system(y, t):
    x = y[:N].astype(np.float64)
    v = y[N:2*N].astype(np.float64)
    phi = y[2*N:3*N].astype(np.float64)

    # Уравнения движения
    dxdt = -v
    dvdt = -(-gamma * (V0**2 - v**2) * v + np.sin(x) - f)

    # Векторизованное вычисление взаимодействий
    interactions = np.zeros(N, dtype=np.float64)
    for i in range(N):
        for j in range(N):
            dphi = phi[j] - phi[i] - alpha
            interactions[i] += G(x[j] - x[i]) * np.sin(dphi)

    # Ручная конкатенация вместо np.concatenate
    result = np.empty(3*N, dtype=np.float64)
    result[:N] = dxdt
    result[N:2*N] = dvdt
    result[2*N:] = interactions

    return result
'''




#@njit(nogil=True, fastmath=True, cache=True)
@njit(fastmath=False)
def G(x):
    """Периодическое ядро взаимодействия"""
    x_mod = (x + L / 2) % L - L / 2
    return kappa * np.cosh(kappa * (np.abs(x_mod) - L / 2)) / (2 * np.sinh(kappa * L / 2))\


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
    dvdt = -(-gamma * (V0  2 - v  2) * v + np.sin(x) - f)

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


#@njit(parallel=True, fastmath=True)
@njit(parallel=True, fastmath=False)
def compute_local_order_parameter_numba(x, phi, bins, L):
    """Векторизованная версия с предварительным вычислением бинов"""
    Z = np.zeros((bins, phi.shape[1]))
    bin_edges = np.linspace(0, L, bins + 1)

    for t_idx in range(phi.shape[1]):

        x_t = x[:, t_idx]
        bin_indices = np.digitize(x_t, bin_edges) - 1

Великий магистр, [16.04.2025 22:05]
bin_indices = np.clip(bin_indices, 0, bins - 1)

        for b in range(bins):
            mask = (bin_indices == b)
            if np.any(mask):
                Z[b, t_idx] = np.abs(np.mean(np.exp(1j * phi[mask, t_idx])))

    return Z

def save_params(params):
    """Сохранить параметры в JSON файл"""
    with open(args.params, 'w') as f:
        json.dump(params, f, indent=2)

def save_to_csv(data, chunk_size=1000):
    """Сохранить данные в CSV с прогресс-баром"""
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    params = {k: v for k, v in globals().items()
              if k in ['N', 'L', 'gamma', 'V0', 'f', 'kappa',
                       'alpha', 't_max', 'dt', 'Q', 'delta']}
    save_params(params)
    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(['time'] + [f'{typ}{i}' for i in range(N) for typ in ['x', 'v', 'phi']])

        with tqdm(total=len(data), desc='Saving to CSV') as pbar:
            for i in range(0, len(data), chunk_size):
                chunk = data[i:i + chunk_size]
                for row in chunk:
                    writer.writerow(row)
                pbar.update(len(chunk))

# Генерация начальных условий
'''
x0 = x0 = np.linspace(-2*L, 2*L, N).astype(np.float64)
v0 = np.random.normal(0, 0.1, N).astype(np.float64)
phi0 = (6  * np.exp(-10*(x0 - L/2)**2) * np.random.uniform(-0.5, 0.5, N)).astype(np.float64)
y0 = np.concatenate((x0, v0, phi0))
'''
np.random.seed(42)
x0 = np.random.uniform(0, L, N)
v0 = np.random.normal(0, 0.1, N)
phi0 = np.random.uniform(-np.pi, np.pi, N)
y0 = np.concatenate((x0, v0, phi0))
# Дальнейший код остается без изменений...



# Интеграция
t = np.arange(0, t_max, dt)
'''
# Параметры
t_total = 1000       # Общее время моделирования
dt_chunk = 100       # Длина интервала для прогресс-бара
t_chunks = np.arange(0, t_total + dt_chunk, dt_chunk)

# Разбиваем время на интервалы
sol = []
info =[]
y_current = y0       # Начальные условия

# Интегрируем по частям с прогресс-баром
with tqdm(total=len(t_chunks)-1, desc="Прогресс") as pbar:
    for i in range(len(t_chunks)-1):
        t_interval = np.linspace(t_chunks[i], t_chunks[i+1], 100)
        sol_chunk = odeint(system, y_current, t_interval)
        sol.append(sol_chunk)
        y_current = sol_chunk[-1]  # Обновляем начальные условия
        pbar.update(1)

# Объединяем результаты
sol = np.vstack(sol)

print

'''
ts = time.perf_counter()
sol, info = odeint(system, y0, t, rtol=1e-6, full_output=True)
te = time.perf_counter()  # Конец замера
ex = te - ts
print("интегрирование завершено за:", ex)

if args.output:
    print("\nСохранение результатов...")
    data = np.hstack((t.reshape(-1, 1), sol))
    save_to_csv(data)



# Постобработка