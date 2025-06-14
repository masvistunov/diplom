import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import utils
def plot_figure4(x_solution, phi_solution,V, potential_wells, time, L, Q, delta,output_dir,changes,r,gam,v0,stepen):
    fig, axes = plt.subplots(5, 1, figsize=(16, 15), gridspec_kw={'height_ratios': [1, 1, 1,1,1]})

# ------------------------------------------------------------
# График 1: |Z_S(x, t)|
# ------------------------------------------------------------
   # x_grid = np.linspace(0, L, len(potential_wells))
#    Z_S = np.zeros((len(x_grid), len(time)))
    print("первый")

    Z_S,Z_M,Z_O = utils.calculateZSAndZM(x_solution.astype(np.float64), phi_solution.astype(np.float64), V.astype(np.float64), time[0:-1], delta, chages= changes,r=r)
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
    axes[0].set(title='Local Static Synchronization', xlabel='Time', ylabel='Position (x)')
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
    axes[3].set(title='Local Moving  Synchronization', xlabel='Time', ylabel='Position (x)')
    im1 = axes[4].imshow(
        np.abs(Z_O),
        aspect='auto',
        extent=[time[0], time[-1], 0, L],
        origin='lower',
        cmap='viridis',
        vmin=0,
        vmax=1
    )
    fig.colorbar(im1, ax=axes[4], label='|Z_O|')
    axes[4].set(title='Local Oscillations Synchronization', xlabel='Time', ylabel='Position (x)')

      # ------------------------------------------------------------
    # График 2: arg R(x_q,t) - Ωt
    # ------------------------------------------------------------
    print("второй")
    R_cluster, omega =  utils.calculate_R_cluster(x_solution, phi_solution, potential_wells, L, Q, time)
    print("2й график")
    phase_diff = ((np.angle(R_cluster) - omega * time[None, :-1])+np.pi) % (2 * np.pi) - np.pi

    # Создаем сетку координат ям [0, L) → нормируем на L → [0, 1)
    y_coords = potential_wells / L
    # Тепловая карта
    im = axes[1].imshow(
        phase_diff,
        aspect='auto',
        extent=[time[0], time[-100], 0, 1],  # X: время, Y: [0, 1)
        origin='lower',
        cmap='inferno',  # Циклическая цветовая карта
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
    print("третий")
    modR = np.abs(R_cluster)
    print("3й")

    im = axes[2].imshow(
        modR,
        aspect='auto',
        extent=[time[0], time[-1], 0, L],  # X: время, Y: координаты [0, L)
        origin='lower',
        cmap='plasma',  # Циклическая цветовая карта
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

    #time_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # С микросекундами

    #filename = os.path.join(output_dir, f"chart_{time_str}.svg")
    filename = os.path.join(output_dir, f"-graf-N{len(x_solution[0])}-Q{Q}-delta{delta}-gamma{gam}-V0{v0}_step{stepen}.svg")
    plt.savefig(filename)
    #plt.show()

def plotglobalParameters(phi,time,v,chagese,output_dir):

    R,R_M,R_O,R_S = utils.globalLocalParam(phi,time,v,chagese)
    # Отрисовка
    plt.figure(figsize=(15, 10))  # Увеличиваем размер фигуры для 4 графиков

    # Создаем 4 субграфика (2x2)
    plt.subplot(2, 2, 1)
    plt.plot(time, np.abs(R), label='$|R_1(t)|$', color='blue')
    plt.xlabel('Время $t$', fontsize=10)
    plt.ylabel('$|R(t)|$', fontsize=10)
    plt.title('Параметр порядка R', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(time, np.abs(R_M), label='$|R_2(t)|$', color='red')
    plt.xlabel('Время $t$', fontsize=10)
    plt.ylabel('$|R(t)|$', fontsize=10)
    plt.title('Параметр порядка R_M', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(time, np.abs(R_S), label='$|R_3(t)|$', color='green')
    plt.xlabel('Время $t$', fontsize=10)
    plt.ylabel('$|R(t)|$', fontsize=10)
    plt.title('Параметр порядка R_S', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(time, np.abs(R_O), label='$|R_4(t)|$', color='purple')
    plt.xlabel('Время $t$', fontsize=10)
    plt.ylabel('$|R(t)|$', fontsize=10)
    plt.title('Параметр порядка R_O', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    time_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # С микросекундами
    filename = os.path.join(output_dir, f"chart_{time_str}.svg")
    plt.savefig(filename)
    plt.show()