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

# ==============================================
# Парсинг аргументов и управление параметрами
# ==============================================
parser = argparse.ArgumentParser(description='Particle system simulation and visualization')
parser.add_argument('--input', type=str, help='Input CSV file with simulation data')
parser.add_argument('--output', type=str, help='Output CSV file to save results')
parser.add_argument('--params', type=str, help='JSON file for parameters')
parser.add_argument('--plot-type', choices=['snapshot', 'order', 'all'], default='all',
                    help='Type of plot to generate (default: all)')
args, _ = parser.parse_known_args()


# ==============================================
# Функции работы с данными
# ==============================================
def save_params(params):
    """Сохранить параметры в JSON файл"""
    with open(args.params, 'w') as f:
        json.dump(params, f, indent=2)


def load_params():
    """Загрузить параметры из JSON файла"""
    with open(args.params) as f:
        return json.load(f)


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


def load_from_csv():
    """Загрузить данные из CSV с прогресс-баром"""
    params = load_params()
    globals().update(params)

    data = []
    with open(args.input, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        next(reader)  # Skip header

        with tqdm(desc='Loading CSV') as pbar:
            for row in reader:
                data.append([float(x) for x in row])
                pbar.update(1)

    t = np.array([x[0] for x in data])
    sol = np.array([x[1:] for x in data]).reshape(len(data), 3 * N)
    return t, sol


# ==============================================
# Основные вычисления
# ==============================================
@njit(fastmath=False)
def G(x, L, kappa):
    """Периодическое ядро взаимодействия"""
    x_mod = (x + L / 2) % L - L / 2
    return kappa * np.cosh(kappa * (np.abs(x_mod) - L / 2)) / (2 * np.sinh(kappa * L / 2))


@njit(parallel=True, fastmath=False)
def system(y, t, L, gamma, V0, f, kappa, alpha):
    """Система ОДУ с параллельными вычислениями"""
    N = len(y) // 3
    x = y[:N]
    v = y[N:2 * N]
    phi = y[2 * N:3 * N]

    dxdt = -v
    dvdt = -(-gamma * (V0 ** 2 - v ** 2) * v + np.sin(x) - f)

    dphidt = np.zeros(N)
    for n in prange(N):
        interaction = 0.0
        for m in range(N):
            dx = x[m] - x[n]
            dphi = phi[m] - phi[n] - alpha
            interaction += G(dx, L, kappa) * np.sin(dphi)
        dphidt[n] = interaction

    return np.concatenate((dxdt, dvdt, dphidt))


# ==============================================
# Визуализация
# ==============================================

def compute_local_order(x, phi, L_value, bins=100):
    """Вычисление локального параметра порядка с бинированием"""
    bin_edges = np.linspace(0, L_value, bins + 1)
    Z = np.zeros((len(x), bins))

    for t in prange(len(x)):
        x_t = x[t] % L_value
        phi_t = phi[t]

        for b in range(bins):
            mask = (x_t >= bin_edges[b]) & (x_t < bin_edges[b + 1])
            if np.any(mask):
                Z[t, b] = np.abs(np.mean(np.exp(1j * phi_t[mask])))

    return Z.mean(axis=0)


def plot_snapshot(sol, L, Q, t):
    """Построение снепшота системы в последний момент времени"""
    plt.figure(figsize=(12, 6))

    N = sol.shape[1] // 3
    x = sol[-1, :N] % L
    phi = sol[-1, 2 * N:3 * N]
    v = sol[-1, N:2 * N]

    stationary = np.abs(v) < 0.01
    moving = ~stationary
    x_wells = np.linspace(0, L, Q, endpoint=False)

    # Стационарные частицы
    plt.scatter(x[stationary], np.angle(np.exp(1j * phi[stationary])) / np.pi,
                c='blue', marker='x', s=50, alpha=0.7, label='Стационарные')

    # Движущиеся частицы
    plt.scatter(x[moving], np.angle(np.exp(1j * phi[moving])) / np.pi,
                c='red', s=30, alpha=0.7, label='Движущиеся')

    # Потенциальные ямы
    plt.scatter(x_wells, np.zeros(Q), c='black', s=40,
                marker='s', label='Ямы потенциала')

    plt.xlabel('Координата, x', fontsize=12)
    plt.ylabel(r'Фаза, $\varphi/\pi$', fontsize=12)
    plt.yticks([-1, -0.5, 0, 0.5, 1],
               ['-π', '-π/2', '0', 'π/2', 'π'])
    plt.ylim(-1.1, 1.1)
    plt.grid(alpha=0.3)
    plt.legend(ncol=3, loc='upper right')
    plt.title(f'Снепшот системы при t = {t[-1]:.1f}', fontsize=14)
    plt.tight_layout()


def plot_order_parameter(sol, L, N):
    """Визуализация параметра порядка"""
    plt.figure(figsize=(12, 6))

    # Вычисление параметра порядка
    x = sol[:, :N]
    phi = sol[:, 2 * N:3 * N]
    Z = compute_local_order(x, phi, L)

    # Сглаживание
    window_size = 100
    Z_smooth = np.convolve(Z, np.ones(window_size) / window_size, mode='valid')

    plt.plot(np.linspace(0, L, len(Z)), Z, alpha=0.3, label='Сырые данные')
    plt.plot(np.linspace(0, L, len(Z_smooth)), Z_smooth,
             linewidth=2, label='Сглаженные данные')

    plt.xlabel('Координата, x', fontsize=12)
    plt.ylabel('Локальный параметр порядка, |Z|', fontsize=12)
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.title('Пространственное распределение параметра порядка', fontsize=14)
    plt.tight_layout()


# ==============================================
# Основной поток выполнения
# ==============================================
if __name__ == '__main__':
    if args.input:
        # Режим визуализации --------------------------------------------------
        print("Загрузка данных...")
        t, sol = load_from_csv()
        params = load_params()
        N = params['N']
        L = params['L']
        Q = params['Q']

        print("\nАнализ данных...")
        if args.plot_type in ['snapshot', 'all']:
            plot_snapshot(sol, L, Q, t)
        if args.plot_type in ['order', 'all']:
            plot_order_parameter(sol, L, N)

        plt.show()

    else:
        # Режим симуляции -----------------------------------------------------
        # Парсинг параметров
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

        # Инициализация параметров
        params = vars(args)
        globals().update(params)

        # Генерация начальных условий
        np.random.seed(42)
        x0 = np.random.uniform(0, L, N)
        v0 = np.random.normal(0, 0.1, N)
        phi0 = np.random.uniform(-np.pi, np.pi, N)
        y0 = np.concatenate((x0, v0, phi0))

        # Интеграция системы
        print("Интеграция системы...")
        t = np.arange(0, t_max, dt)
        sol = odeint(system, y0, t, args=(L, gamma, V0, f, kappa, alpha), rtol=1e-6)

        # Сохранение результатов
        if args.output:
            print("\nСохранение результатов...")
            data = np.hstack((t.reshape(-1, 1), sol))
            save_to_csv(data)

        # Построение быстрого предпросмотра
#        plot_snapshot(sol, L, Q, t)
#        plot_order_parameter(sol, L, N)
#        plt.show()