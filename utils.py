import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
L = 1.0

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
    #omega = np.mean(calculateDotPhi(phi_solution,time))

    print("omega = ", omega)
    return R_cluster, omega

def calculate_R_cluster_new(x_solution, phi_solution, potential_wells, L, Q, time):
    radius = 0.5 * L / Q
    time_len = len(time)-1
    R_cluster = compute_R_cluster_numba(
        x_solution.astype(np.float64),
        phi_solution.astype(np.float64),
        potential_wells.astype(np.float64),
        radius, Q, time_len
    )


@njit(parallel= False, fastmath=False)
def s_neighborhood_matrix_np(x: np.ndarray, y: np.ndarray, s: float) -> np.ndarray:
    x_norm = x % 1
    y_norm = y % 1

    # Создаем матрицу попарных расстояний между x и y
    diff = np.abs(x_norm[:, None] - y_norm)  # shape: (len(x), len(y))
    ring_dist = np.minimum(diff, 1 - diff)

    # Проверяем условие вхождения в окрестность
    return ring_dist < s
@njit(parallel= False, fastmath=False)
def calculateZSAndZM(x_sol,phi_sol,xDot_sol,time,delta, chages,r):
    x_grid = np.linspace(0,L,1000)
    Z_S = np.zeros((len(x_grid),(len(time))),dtype=np.complex128)
    Z_M = np.zeros((len(x_grid),(len(time))),dtype=np.complex128)
    Z_O = np.zeros((len(x_grid), (len(time))), dtype=np.complex128)
    for t in range (len(time)):
        mask =  s_neighborhood_matrix_np(x_grid,x_sol[t,:],delta)
        if (t == 100000):
            print(mask)
        for x in range(len(x_grid)):
            mask_S= (mask[x] & (np.abs(xDot_sol[t,:]) <= 0.01))
            mask_M = (mask[x] & (chages < r))
            mask_O = (mask[x] & (chages >= r))
            ''' считаем что глобально стационарных частиц нет, а остальные частицы мы разделяем на глобально колебательные и глобально вращательные( пролетные)'''
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
            if (np.any(mask_O)):
                Z_O[x,t] = np.mean(np.exp(1j * phi_sol[t,mask_O]))
            else:
                Z_O[x,t] = 0.0
        if ((t % 10000) == 0):
            print(t)
    return Z_S,Z_M,Z_O
def calculateDotPhi(phi,time):
    dot_phi = np.zeros((len(phi[0]),(len(time)-100)))
    dt = time[1] - time[0]
    for t in range(len(time)-1):
        dot_phi[:,t] = (phi[t+1] - phi[t])/dt
    return dot_phi
def globalLocalParam(phi,time, v,chages):
    globalR = np.zeros(len(time),dtype = np.complex128)
    globalR_S = np.zeros(len(time), dtype=np.complex128)
    globalR_M = np.zeros(len(time), dtype=np.complex128)
    globalR_O = np.zeros(len(time), dtype=np.complex128)
    mask_o = chages >= 100
    mask_m = ~mask_o
    for t in range(len(time)):
        mask_s = np.abs(v[t, :]) <= 0.01

        globalR[t] = np.mean(np.exp(1j * phi[t,:]))
        if(np.any(mask_s)):
            globalR_S[t] = np.mean(np.exp(1j * phi[t, mask_s]))
        else:
            globalR_S[t] = 0.0
        if(np.any(mask_m)):
            globalR_M[t] = np.mean(np.exp(1j * phi[t, mask_m]))
        else:
            globalR_M[t] = 0.0
        if(np.any(mask_o)):
            globalR_O[t] = np.mean(np.exp(1j * phi[t, mask_o]))
        else:
            globalR_O[t] = 0.0
    return globalR,globalR_M,globalR_O,globalR_S

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
            else:
                R_cluster[q,ti] = 0.0
    return R_cluster

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
        increasing = np.all(diffs >= -0.1)
        decreasing = np.all(diffs <= 0.1)

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