import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
def snapshot(x,phases,stationary,moving,x_wells,output_dir,t,t_idx,osc,gam,v_0):
    plt.figure(figsize=(10, 6))

    # Стационарные частицы (синие крестики)
    """
    plt.scatter(
        x[stationary],
        np.angle(np.exp(1j * phases[stationary])) / np.pi,  # Исправлены скобки
        c='blue', marker='x', s=50, label='Стационарные', alpha=0.7
    )
    """
    # Движущиеся частицы (красные кружки)
    plt.scatter(
        x[moving],
        np.angle(np.exp(1j * phases[moving])) / np.pi,
        c='red', s=30, label='Движущиеся', alpha=0.7
    )
    # Движущиеся частицы (красные кружки)
    plt.scatter(
        x[osc],
        np.angle(np.exp(1j * phases[osc])) / np.pi,
        c='blue', marker='x', s=50, label='Колеблющиеся', alpha=0.6
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
    #plt.title(f'Снепшот фаз частиц при $t = 5000', fontsize=14)  # Исправлено t_plot на t
    plt.tight_layout()
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # С микросекундами
    filename = os.path.join(output_dir, f"chart_{time_str},N{len(x)}_gam{gam}_v{v_0}.svg")
    plt.savefig(filename)
    #plt.show()