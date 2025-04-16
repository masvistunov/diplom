import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import time
from numba import njit, prange


start_time = time.perf_counter()  # Начало замера

# Параметры
N = 50
L = 1.0
gamma = 0.3
V0 = 0.3
f = 0.8
kappa = 5.2
alpha = 1.457
t_max = 100
dt = 0.01
Q = 25
delta = 15




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
@njit(parallel=True, fastmath=False)
def compute_local_order_parameter_numba(x, phi, bins, L):
    """Векторизованная версия с предварительным вычислением бинов"""
    Z = np.zeros((bins, phi.shape[1]))
    bin_edges = np.linspace(0, L, bins + 1)

    for t_idx in range(phi.shape[1]):

        x_t = x[:, t_idx]
        bin_indices = np.digitize(x_t, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, bins - 1)

        for b in range(bins):
            mask = (bin_indices == b)
            if np.any(mask):
                Z[b, t_idx] = np.abs(np.mean(np.exp(1j * phi[mask, t_idx])))

    return Z


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
print(sol[:,N])
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
    plt.show()


###########
def analyze_results(solution, N, L, delta, bins=1000):
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

    return Z_oscillatory, Z_rotational


##########
def plt2(solution, Z_oscillatory, Z_rotational, t_span, L):
    print("Исследование для количества частиц N = ", N, "\n")
    print("Q = ", Q, "\n")
    print("delta = ", delta, "\n")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 20), sharex=True)

    x_solution = sol[:N, :] % L
    phi_solution = sol[2 * N:3 * N, :]

    vmin = min(np.min(Z_oscillatory), np.min(Z_rotational))
    vmax = max(np.max(Z_oscillatory), np.max(Z_rotational))

    # Колебательная часть
    im1 = ax1.imshow(Z_oscillatory, aspect='auto',
                     extent=[t_span[0], t_span[1], 0, L],
                     origin='lower', cmap='viridis',
                     vmin=vmin, vmax=vmax)
    ax1.set_title('Oscillatory Particles')
    ax1.set_ylabel('Position x')
    fig.colorbar(im1, ax=ax1, label='|Z(x, t)|')

    # Вращательная часть
    im2 = ax2.imshow(Z_rotational, aspect='auto',
                     extent=[t_span[0], t_span[1], 0, L],
                     origin='lower', cmap='plasma',
                     vmin=vmin, vmax=vmax)
    ax2.set_title('Rotational Particles')
    ax2.set_xlabel('Time t')
    ax2.set_ylabel('Position x')
    fig.colorbar(im2, ax=ax2, label='|Z(x, t)|')
    plt.tight_layout()
    return fig


plt1()
Z_osc, Z_rot = analyze_results(sol, N, L, delta)
plt2(sol, Z_osc, Z_rot, (0,t_max), L)
plt.show()
end_time = time.perf_counter()  # Конец замера
execution_time = end_time - start_time
# Вывод результатов
print('/n')
print(f"Время выполнения: {execution_time:.4f} секунд")