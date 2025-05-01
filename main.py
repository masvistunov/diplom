import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import time
from numba import njit, prange
from datetime import datetime
import os

start_time = time.perf_counter()  # Начало замера

# Параметры
N = 10
L = 1.0
gamma = 1.5
V0 = 1.0
f = 0.8
kappa = 5.2
alpha = 1.457
t_max = 100
dt = 0.01
Q =  65
delta = 0.05
output_dir = f"C:/work/"



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
    dvdt = -(-gamma * (V0 ** 2 - v ** 2) * v + np.sin(x) - f)

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
#@njit(parallel=True, fastmath=False)

def calculate_Z_over_time(x_solution, phi_solution, potential_wells, delta):
    """
    Вычисляет Z(t) для стационарных и пролетных частиц в каждый момент времени.

    Параметры:
    x_solution : array (N, T) - координаты частиц
    phi_solution : array (N, T) - фазы частиц
    potential_wells : array (Q,) - координаты ям
    delta : float - размер окрестности ямы

    Возвращает:
    Z_S : array (T,) - параметр порядка стационарных частиц
    Z_M : array (T,) - параметр порядка пролетных частиц
    """
    N, T = x_solution.shape
    Z_S = np.zeros(T, dtype=np.complex64)
    Z_M = np.zeros(T, dtype=np.complex64)

    for ti in range(T):
        # Классификация частиц в момент времени ti
        stationary_mask, flying_mask = classify_particles(
            x_solution[ti, :],
            potential_wells,
            delta
        )

        # Вычисление Z_S(ti) и Z_M(ti)
        Z_S[ti] = calculate_Z(phi_solution[ti, :], stationary_mask)
        Z_M[ti] = calculate_Z(phi_solution[ti, :], flying_mask)

    return Z_S, Z_M

def classify_particles(x, potential_wells, delta):

    stationary_mask = np.zeros_like(x, dtype=bool)
    for xq in potential_wells:
        # Учет периодичности кольца [0, 1)
        lower = (xq - delta / 2) % 1
        upper = (xq + delta / 2) % 1
        if lower < upper:
            mask = (x >= lower) & (x <= upper)
        else:
            mask = (x >= lower) | (x <= upper)
        stationary_mask |= mask
    flying_mask = ~stationary_mask
    return stationary_mask, flying_mask
#@njit(parallel=True, fastmath=False)
def clcZ(x_solution,phi_solution,time):
    delta_abs = delta * L
    x_grid = np.linspace(0, L, 1000)
    Z_S = np.zeros((len(x_grid), len(time)))
    """
    stationary_mask = np.zeros_like(x_solution, dtype=bool)
    for ti, t in enumerate(time):
        for xq in potential_wells:
        # Учет периодичности кольца [0, 1)
            lower = (xq - delta / 2) % 1
            upper = (xq + delta / 2) % 1
            if lower < upper:
                mask = (x_solution[ti,xq] >= lower) & (x_solution[ti,xq] <= upper)
            else:
                mask = (x_solution[ti,xq] >= lower) | (x_solution[ti,xq] <= upper)
        stationary_mask |= mask
        """
    # Расчет локальной синхронизации
    for ti, t in enumerate(time):
        for xi, x in enumerate(x_grid):
            mask = (x_solution[ti, :] >= x - delta_abs) & (x_solution[ti, :] <= x + delta_abs)
            if mask.any():
                Z_S[xi, ti] = np.abs(np.mean(np.exp(1j * phi_solution[ti, mask])))
    return Z_S
#@njit(parallel= False, fastmath=False)

@njit(parallel= False, fastmath=False)
def s_neighborhood_matrix_np(x: np.ndarray, y: np.ndarray, s: float) -> np.ndarray:
    x_norm = x % 1
    y_norm = y % 1

    # Создаем матрицу попарных расстояний между x и y
    diff = np.abs(x_norm[:, None] - y_norm)  # shape: (len(x), len(y))
    ring_dist = np.minimum(diff, 1 - diff)

    # Проверяем условие вхождения в окрестность
    return ring_dist < s

def calculateZSAndZM(x_sol,phi_sol,xDot_sol,time,delta):
    x_grid = np.linspace(0,L,10)
    Z_S = np.zeros((len(x_grid),(len(time))),dtype=np.complex128)
    Z_M = np.zeros((len(x_grid),(len(time))),dtype=np.complex128)
    for t in range (len(time)):
        mask =  s_neighborhood_matrix_np(x_grid,x_sol[t,:],delta)
        for x in range(len(x_grid)):
            mask_S = (mask[x] & (np.abs(xDot_sol[t,:])<= 0.01))
            mask_M = (mask[x] & (np.abs(xDot_sol[t,:]) > 0.01))
            #mask_S = ((x_sol[t,:] >= (x - delta)%1) & (x_sol[t,:] <= (x + delta)%1) & (np.abs(xDot_sol[t,:])<= 0.01))
            #mask_M = ((x_sol[t, :] >= (x - delta)%1) & (x_sol[t, :] <= (x + delta)%1) & (np.abs(xDot_sol[t,:]) > 0.01))
            if np.any(mask_S):
                Z_S[x,t] = np.mean(np.exp(1j * phi_sol[t,mask_S]))
            else:
                Z_S[x,t] = 0.0
            if np.any(mask_M):
                Z_M[x,t] = np.mean(np.exp(1j * phi_sol[t,mask_M]))
            else:
                Z_M[x,t] = 0.0
    return Z_S,Z_M


def calculate_Z(phi, mask):
    if np.sum(mask) == 0:
        return 0.0  # Если группа пуста
    return np.mean(np.exp(1j * phi[mask]))

def plot_figure4(x_solution, phi_solution,V, potential_wells, time, L, Q, delta):
    fig, axes = plt.subplots(4, 1, figsize=(16, 15), gridspec_kw={'height_ratios': [1, 1, 1,1]})
    print("зашли в построение графиков")
# ------------------------------------------------------------
# График 1: |Z_S(x, t)|
# ------------------------------------------------------------
   # x_grid = np.linspace(0, L, len(potential_wells))
#    Z_S = np.zeros((len(x_grid), len(time)))
    Z_S,Z_M = calculateZSAndZM(x_solution.astype(np.float64),phi_solution.astype(np.float64),V.astype(np.float64),time,delta)
# Отрисовка
    im1 = axes[0].imshow(
        np.abs(Z_S),
     aspect='auto',
        extent=[time[0], time[-1], 0, L],
    origin='lower',
    cmap='viridis',
    vmin=0,
    vmax=1
    )
    fig.colorbar(im1, ax=axes[0], label='|Z_S|')
    axes[0].set(title='Local Synchronization', xlabel='Time', ylabel='Position (x)')
    im1 = axes[3].imshow(
        np.abs(Z_M),
     aspect='auto',
        extent=[time[0], time[-1], 0, L],
    origin='lower',
    cmap='viridis',
    vmin=0,
    vmax=1
    )
    fig.colorbar(im1, ax=axes[3], label='|Z_M|')
    axes[3].set(title='Local Synchronization', xlabel='Time', ylabel='Position (x)')

    # ------------------------------------------------------------
    # График 2: arg R(x_q,t) - Ωt
    # ------------------------------------------------------------
    R_cluster, omega =  calculate_R_cluster(x_solution, phi_solution, potential_wells, L, Q, time)
    print("2й график")
    phase_diff = ((np.angle(R_cluster) - omega * time[None, :])+np.pi) % (2 * np.pi) - np.pi

    # Создаем сетку координат ям [0, L) → нормируем на L → [0, 1)
    y_coords = potential_wells / L
    # Тепловая карта
    im = axes[1].imshow(
        phase_diff,
        aspect='auto',
        extent=[time[0], time[-1], 0, 1],  # X: время, Y: [0, 1)
        origin='lower',
        cmap='hsv',  # Циклическая цветовая карта
        vmin=-np.pi,
        vmax=np.pi,
        interpolation='nearest'
    )

    # Настройка цветовой шкалы
    cbar = fig.colorbar(im, ax=axes[1], label='arg R(x, t) - Ωt')
    cbar.set_ticks([-np.pi, 0, np.pi])
    cbar.set_ticklabels(['-π', '0', 'π'])

    # Подписи осей
    axes[1].set(
        xlabel='Time',
        ylabel='Position (x)',
        title='Phase Deviation (arg R(x, t) - Ωt)',
        yticks=np.linspace(0, 1, 5)  # Метки: 0, 0.25, 0.5, 0.75, 1
    )

    # ------------------------------------------------------------
    # График 3: |R(x_q,t)|
    # ------------------------------------------------------------
    modR = np.abs(R_cluster)  # (Q, T)
    print("3й")

    im = axes[2].imshow(
        modR,
        aspect='auto',
        extent=[time[0], time[-1], 0, L],  # X: время, Y: координаты [0, L)
        origin='lower',
        cmap='hsv',  # Циклическая цветовая карта
        vmin=0,
        vmax=1
    )

    # Настройка цветовой шкалы
    cbar = fig.colorbar(im, ax=axes[2], label=' |R(x, t)|')
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['0', '1'])

    # Подписи осей
    axes[2].set(
        xlabel='Time',
        ylabel='Position (x)',
        title='Phase of clusters |R(x, t)|)',
        ylim=(0, L)  # Ось Y от 0 до L (например, 1)
    )

    # Общие настройки
    plt.tight_layout()

    time_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # С микросекундами
    filename = os.path.join(output_dir, f"chart_{time_str}.svg")
    plt.savefig(filename)
    plt.show()
@njit(parallel=True, fastmath=False)
def compute_R_cluster_numba(x_solution, phi_solution, potential_wells, radius, Q, time_len):
    R_cluster = np.zeros((Q, time_len), dtype=np.complex128)
    for ti in prange(time_len):
        mask = s_neighborhood_matrix_np(potential_wells, x_solution[ti, :], radius)

        for q in prange(Q) :
            xq = potential_wells[q]
            #mask = np.abs(x_solution[ti, :] - xq) <= radius
            #mask  = s_neighborhood_matrix_np(potential_wells,x_solution[ti,:],radius)
            if np.any(mask[q]):
                R_cluster[q, ti] = np.mean(np.exp(1j * phi_solution[ti, mask[q]]))
            #else:
                #R_cluster[q,t] = 0.0
    return R_cluster


def calculate_R_cluster(x_solution, phi_solution, potential_wells, L, Q, time):
    radius = 0.5 * L / Q
    time_len = len(time)

    # Вычисление R_cluster с параллелизацией через numba
    R_cluster = compute_R_cluster_numba(
        x_solution.astype(np.float64),
        phi_solution.astype(np.float64),
        potential_wells.astype(np.float64),
        radius, Q, time_len
    )

    # Расчет omega (можно тоже ускорить через numba)
    omega_per_cluster = np.zeros(Q)
    for q in range(Q):
        phi_avg = np.arctan2(np.sin(np.unwrap(np.angle(R_cluster[q, :]))),
                             np.cos(np.unwrap(np.angle(R_cluster[q, :]))))
        omega_per_cluster[q] = np.mean(np.gradient(phi_avg, time))

    omega = np.mean(omega_per_cluster)
    print("omega = ", omega)
    return R_cluster, omega
"""def calculate_R_cluster(x_solution, phi_solution, potential_wells, L, Q, time):
    R_cluster = np.zeros((Q, len(time)), dtype=complex)
    radius = 0.5 * L / Q
    # идем по каждой ПЯ
    for q in range(Q):
        xq = potential_wells[q] # координата ПЯ
        for ti in range(len(time)):
            mask = np.abs(x_solution[ti, :] - xq) <= radius # проверяем назходится ли частица в потенциальной яме
            if mask.any():
                R_cluster[q, ti] = np.mean(np.exp(1j * phi_solution[ti, mask]))

    # Расчет средней частоты
    omega_per_cluster = np.zeros(Q)
    for q in range(Q):
        #phi_avg = np.unwrap(np.angle(R_cluster[q, :]))
        phi_avg = np.arctan2(np.sin(np.unwrap(np.angle(R_cluster[q, :]))), np.cos(np.unwrap(np.angle(R_cluster[q, :]))))
        omega_per_cluster[q] = np.mean(np.gradient(phi_avg, time))
    omega = np.mean(omega_per_cluster)
    print("omega = ", omega)
    return R_cluster, omega"""

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

ts = time.perf_counter()
sol = odeint(system, y0, t, rtol=1e-6)
te = time.perf_counter()  # Конец замера
ex = te - ts
print("интегрирование завершено за:", ex)
# Постобработка
stationary = np.abs(sol[:, N:2 * N]) < 0.01  # маска стационарных частиц

t_idx = -1  # Индекс момента времени

# ------------------------------------------------------------
# 1. Подготовка данных
# ------------------------------------------------------------
# Выборка данных для момента времени t_idx
x = sol[t_idx, :N] % L  # Координаты частиц
phases = sol[t_idx, 2 * N:3 * N]  # Фазы частиц
v = sol[t_idx, N:2 * N]  # Скорости частиц
x_solution = sol[:,:N] %L
phi_solution = sol[:,2*N:3 *N]
v_solution =  sol[:,N:2*N]
print(type(v_solution))
# Определение стационарных и движущихся частиц
stationary = np.abs(v) < 0.01

moving = ~stationary

# Координаты потенциальных ям
x_wells = np.linspace(0, L, Q, endpoint=False)


# ------------------------------------------------------------
# 2. Построение графика
# ------------------------------------------------------------
def plt1():
    plt.figure(figsize=(10, 6))

    # Стационарные частицы (синие крестики)
    plt.scatter(
        x[stationary],
        np.angle(np.exp(1j * phases[stationary])) / np.pi,  # Исправлены скобки
        c='blue', marker='x', s=50, label='Стационарные', alpha=0.7
    )

    # Движущиеся частицы (красные кружки)
    plt.scatter(
        x[moving],
        np.angle(np.exp(1j * phases[moving])) / np.pi,
        c='red', s=30, label='Движущиеся', alpha=0.7
    )

    # Потенциальные ямы (черные маркеры)
    plt.scatter(
        x_wells,
        np.zeros_like(x_wells),
        c='black', s=40, marker='s', label='Потенциальные ямы'
    )

    # Настройки графика
    plt.xlabel('Координата, $x$', fontsize=12)
    plt.ylabel(r'Фаза, $\varphi_n/\pi$', fontsize=12)  # Добавлен префикс r
    plt.yticks([-1.0, -0.5, 0.0, 0.5, 1.0],
               ['-pi', '-pi/2', '0', 'pi/2', 'pi'])
    plt.ylim(-1.2, 1.2)
    plt.grid(alpha=0.3)
    plt.legend(loc='upper right', ncol=3)

    plt.title(f'Снепшот фаз частиц при $t = {t[t_idx]:.1f}$', fontsize=14)  # Исправлено t_plot на t
    plt.tight_layout()
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # С микросекундами
    filename = os.path.join(output_dir, f"chart_{time_str}.svg")
    plt.savefig(filename)
    plt.show()


###########
"""def analyze_results(solution, N, L, delta, bins=1000):
    x_solution = sol[:,:N] % L  # Координаты частиц
    phi_solution = sol[:, 2 * N:3 * N]  # Фазы частиц
    velocity = sol[:, N:2 * N]  # Скорости частиц

    max_velocity = np.max(np.abs(velocity), axis=1)
    oscillatory_mask = max_velocity < 0.5
    rotational_mask = ~oscillatory_mask

    x_osc = x_solution[oscillatory_mask]
    phi_osc = phi_solution[oscillatory_mask]
    x_rot = x_solution[rotational_mask]
    phi_rot = phi_solution[rotational_mask]

    Z_oscillatory = compute_local_order_parameter_numba(x_osc, phi_osc, bins, L)
    Z_rotational = compute_local_order_parameter_numba(x_rot, phi_rot, bins, L)

    return Z_oscillatory, Z_rotational"""

plt1()

# Предположим данные уже загружены:
# x_solution, phi_solution, potential_wells, time, L, Q
potential_wells = np.linspace(0, L, Q, endpoint=False)

plot_figure4(
    x_solution=x_solution,
    phi_solution=phi_solution,
    V= v_solution,
    potential_wells =potential_wells,
    time=t,
    L=L,
    Q=Q,
    delta = delta
    )



plt.show()
end_time = time.perf_counter()  # Конец замера
execution_time = end_time - start_time
# Вывод результатов
print('/n')
print(f"Время выполнения: {execution_time:.4f} секунд")