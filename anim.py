import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.animation import FuncAnimation, PillowWriter
from datetime import datetime
import os
def anim(x_solution,phi_solution,L,N,t_max):
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