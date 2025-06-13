import numpy as np
import snapshot
import globalAndLocalgraf
import anim
import utils
import argparse

# Чтение аргументов командной строки
parser = argparse.ArgumentParser(description='Solve the ODE system.')
parser.add_argument('--N', type=int, default=50, help='Number of particles')
parser.add_argument('--t_max', type=int, default=2500, help='Maximum simulation time')
parser.add_argument('--Q', type=int, default=5, help='Parameter Q')
parser.add_argument('--delta', type=float, default=0.05, help='Parameter delta')
parser.add_argument('--dt',type=float,default=0.01)
parser.add_argument('--gam',type=float,default=1.5)
parser.add_argument('--v_0',type=float,default=0.12)
parser.add_argument('--stepen',type=int,default=3)
args = parser.parse_args()


# Параметры

N = args.N
t_max = args.t_max
Q = args.Q
delta = args.delta
dt = args.dt
gam = args.gam
v_0=args.v_0
stepen = args.stepen


#N = 100
L = 1.0
f = 0.8
kappa = 5.2
alpha = 1.457
"""
t_max = 5000
dt = 0.01
delta = 0.05
gam=0.3
v_0=0.12
Q = 5
"""

gamma = gam * Q ** (-3 / 2)
V0 = v_0 * Q ** (1 / 2)
output_dir = f"./work"
#sol = np.load(f'---solution_N{N}_T{t_max}_Q{Q}_gamma{gam}_V0{v_0}.npy')
sol = np.load(f'solution_N{N}_T{t_max}_Q{Q}.npy')
t = np.arange(0, t_max+1 , dt)
t_idx = -1
x = sol[t_idx, :N] % L  # Координаты частиц
phases = sol[t_idx, 2 * N:3 * N]  # Фазы частиц
v = sol[t_idx, N:2 * N]  # Скорости частиц
x_solution = sol[:,:N] %L
phi_solution = sol[:,2*N:3 *N]
v_solution =  sol[:,N:2*N]
print(len(x_solution[0]))
# Определение стационарных и движущихся частиц
stationary = np.abs(v) < 0.01
#moving = ~stationary
# Нормализация фазы в диапазон [-π, π]
phi_solution = np.mod(phi_solution + np.pi, 2 * np.pi) - np.pi

# Координаты потенциальных ям
x_wells = np.linspace(0, L, Q, endpoint=False)
chagese =  utils.count_monotonicity_changes(sol[:,:N])
r = 300
moving = chagese <=r
osc = chagese > r

#anim.anim(x_solution,phi_solution,L,N,t_max)
snapshot.snapshot(x,phases,stationary,moving,x_wells,output_dir,t,t_idx,osc,gam, v_0)
globalAndLocalgraf.plot_figure4(x_solution,phi_solution,v_solution,x_wells,t,L,Q,delta,output_dir,chagese,r,gamma,V0)
#globalAndLocalgraf.plotglobalParameters(phi_solution,t,v_solution,chagese,output_dir)
#

#res = utils.check_monotonicity_2d(phi_solution,start_row)
print("конец")
