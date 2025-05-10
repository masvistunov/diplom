import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import time
from numba import njit, prange
from datetime import datetime
import os
from matplotlib import rc
from matplotlib.animation import FuncAnimation, PillowWriter
start_time = time.perf_counter()  # Начало замера

# Параметры
N = 51
L = 1.0
gamma = 0.3
V0 = 0.3
f = 0.8
kappa = 5.2
alpha = 1.457
t_max = 5000
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
    x_grid = np.linspace(0,L,1000)
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
def calculateDotPhi(phi,time):
    dot_phi = np.zeros((len(phi),len(time)-1))
    dt = time[1] - time[0]
    for t in range(len(time)-1):
        dot_phi = (phi[t+1] - phi[t])/dt
    return dot_phi
def globalLocalParam(phi,time):
    R = np.zeros(len(time),dtype = np.complex128)
    for t in range(len(time)):
        R[t] = np.mean(np.exp(1j * phi[t,:]))
    plt.figure(figsize=(10, 5))
    plt.plot(time, np.abs(R), label='Глобальный параметр порядка $|R(t)|$', color='blue')
    plt.xlabel('Время $t$', fontsize=12)
    plt.ylabel('$|R(t)|$', fontsize=12)
    plt.title('Динамика глобального параметра порядка', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()
def clasification(phi,time):
    epsilon = 1e-3
    dot_phi = calculateDotPhi(phi,time)
    stat_mask = np.zeros((len(dot_phi),len(time)))
    wavering_mask = np.zeros((len(dot_phi),len(time)))
    rotating_mask = np.zeros((len(dot_phi),len(time)))
    delta_phi = np.zeros(len(dot_phi))
    for t in range(len(time)):
         delta_phi = np.max(phi[t,:]) - np.min(phi[t,:])
         wavering_mask[:,t] = np.where((np.abs(np.mean(dot_phi[t])) <= epsilon) & (delta_phi > 0.1))[0]

    return
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
    Z_S,Z_M = calculateZSAndZM(x_solution.astype(np.float64),phi_solution.astype(np.float64),V.astype(np.float64),time[0:-2],delta)
# Отрисовка
    im1 = axes[0].imshow(
        np.abs(Z_S),
     aspect='auto',
        extent=[time[0], time[-2], 0, L],
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
        extent=[time[0], time[-2], 0, L],
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
    R_cluster, omega =  calculate_R_cluster(x_solution, phi_solution, potential_wells, L, Q, time[0:-2])
    print("2й график")
    phase_diff = ((np.angle(R_cluster) - omega * time[None, :])+np.pi) % (2 * np.pi) - np.pi

    # Создаем сетку координат ям [0, L) → нормируем на L → [0, 1)
    y_coords = potential_wells / L
    # Тепловая карта
    im = axes[1].imshow(
        phase_diff,
        aspect='auto',
        extent=[time[0], time[-2], 0, 1],  # X: время, Y: [0, 1)
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
        extent=[time[0], time[-2], 0, L],  # X: время, Y: координаты [0, L)
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


import numpy as np

def check_monotonicity_2d(data, start_row=0, strict=False):
    """
    Проверяет монотонность для каждого столбца в 2D массиве и находит точку нарушения.

    Параметры:
        data (np.ndarray): 2D массив (время по строкам, частицы по столбцам)
        start_row (int): С какой строки начинать проверку
        strict (bool): Проверять строгую монотонность

    Возвращает:
        dict: {
            'is_monotonic': np.array[bool],  # Является ли монотонным весь интервал
            'type': np.array[str],          # Тип монотонности
            'break_point': np.array[int]    # Индекс первого нарушения (-1 если нет)
        }
    """
    if start_row >= data.shape[0]:
        raise ValueError("start_row превышает количество строк")

    results = {
        'is_monotonic': np.ones(data.shape[1], dtype=bool),
        'type': np.full(data.shape[1], 'none', dtype=object),
        'break_point': np.full(data.shape[1], -1, dtype=int)
    }

    for col in range(data.shape[1]):
        column_data = data[start_row:, col]
        if len(column_data) < 2:
            results['type'][col] = 'constant'
            continue

        diffs = np.diff(column_data)

        # Проверяем все возможные случаи монотонности
        increasing = np.all(diffs >= 0)
        decreasing = np.all(diffs <= 0)

        if increasing:
            mono_type = 'increasing'
            expected_sign = 1
            if strict and np.any(diffs == 0):
                results['is_monotonic'][col] = False
                mono_type = 'non-strictly increasing'
        elif decreasing:
            mono_type = 'decreasing'
            expected_sign = -1
            if strict and np.any(diffs == 0):
                results['is_monotonic'][col] = False
                mono_type = 'non-strictly decreasing'
        else:
            # Находим первое нарушение для неупорядоченных данных
            results['is_monotonic'][col] = False
            results['type'][col] = 'none'

            # Сначала пытаемся определить общий тренд
            pos_diff = np.sum(diffs > 0)
            neg_diff = np.sum(diffs < 0)

            if pos_diff > neg_diff:
                expected_sign = 1  # Преимущественно возрастает
                mono_type = 'mostly increasing'
            else:
                expected_sign = -1  # Преимущественно убывает
                mono_type = 'mostly decreasing'

            # Ищем точку нарушения основного тренда
            for i in range(1, len(column_data)):
                if (column_data[i] - column_data[i - 1]) * expected_sign < 0:
                    results['break_point'][col] = i + start_row
                    break
            continue

        # Для монотонных данных проверяем строгость
        if strict:
            if mono_type.startswith('increasing') and np.any(diffs == 0):
                mono_type = 'non-strictly increasing'
            elif mono_type.startswith('decreasing') and np.any(diffs == 0):
                mono_type = 'non-strictly decreasing'
            else:
                mono_type = 'strictly ' + mono_type

        results['type'][col] = mono_type

        # Для монотонных данных проверяем строгость на всем интервале
        if strict:
            if ('increasing' in mono_type and np.any(diffs == 0)) or \
                    ('decreasing' in mono_type and np.any(diffs == 0)):
                results['is_monotonic'][col] = False
                # Находим первую точку, где нарушается строгость
                for i in range(len(diffs)):
                    if diffs[i] == 0:
                        results['break_point'][col] = i + 1 + start_row
                        break

    return results


def count_monotonicity_changes(x_sol):
    """
    Подсчитывает количество изменений монотонности для каждой частицы

    Параметры:
    x_sol : ndarray
        2D массив, где x_sol[t,i] - координата i-й частицы в момент времени t

    Возвращает:
    changes : ndarray
        Массив с количеством изменений монотонности для каждой частицы
    """
    # Вычисляем разности между соседними временными точками
    diffs = np.diff(x_sol, axis=0)

    # Определяем знаки разностей (1 для возрастания, -1 для убывания)
    signs = np.sign(diffs)

    # Находим изменения знаков (игнорируем нули - когда разность равна 0)
    sign_changes = (signs[:-1] * signs[1:]) < 0

    # Суммируем изменения для каждой частицы
    changes = np.sum(sign_changes, axis=0)

    return changes



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
t = np.arange(0, t_max+1 , dt)

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

changes = count_monotonicity_changes(sol[:,:N])

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
res = check_monotonicity_2d(phi_solution,int(t_max/2))
moving = ~stationary

# Нормализация фазы в диапазон [-π, π]
phi_solution = np.mod(phi_solution + np.pi, 2 * np.pi) - np.pi





###################

plt.rcParams['animation.embed_limit'] = 60

# Создание фигуры и осей
fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(x_solution[0, :], np.arange(N), c=phi_solution[0, :], cmap='hsv', s=50)

# Настройка осей
ax.set_xlim(0, L)
ax.set_ylim(-1, N)
ax.set_xlabel('Position (x)')
ax.set_ylabel('Particle Index')
ax.set_title('Snapshots of Particle Coordinates and Phases Over Time')

# Цветовая шкала
cb = plt.colorbar(sc, ax=ax)
cb.set_label('Phase (radians)')
cb.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
cb.ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])

# Функция для обновления кадров анимации
def update(frame):
    sc.set_offsets(np.c_[x_solution[frame, :], np.arange(N)])
    sc.set_array(phi_solution[frame,   :])
    return sc,

# Создание анимации
print(x_solution.shape[1])
ani = FuncAnimation(fig, update, frames=int(t_max/2), interval=50, blit=True)

#ani.save('output.mp4', fps=30, dpi=300)  # 5000 кадров → ~2.78 минуты
ani.save('animation.gif', writer='pillow', fps=30)

# Показать анимацию в отдельном окне
plt.show()
#########################
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

    plt.title(f'Снепшот фаз частиц при $t = {t[t_idx-1]:.1f}$', fontsize=14)  # Исправлено t_plot на t
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
#clasification(phi_solution,t)
plt1()
globalLocalParam(phi_solution,t)
# Предположим данные уже загружены:
# x_solution, phi_solution, potential_wells, time, L, Q
potential_wells = np.linspace(0, L, Q, endpoint=False)
"""
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

"""

#plt.show()
end_time = time.perf_counter()  # Конец замера
execution_time = end_time - start_time
# Вывод результатов
print('/n')
print(f"Время выполнения: {execution_time:.4f} секунд")