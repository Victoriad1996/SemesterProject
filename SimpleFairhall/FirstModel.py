import matplotlib.pyplot as plt
from functions import*
from brian2 import*


start_scope()


rows, cols = 20, 20

# Recurrent weight from PC to PC
w_PC = 0.5
# Weight PC to INH
w_PCINH = 0.7
# Weight INH to PC
w_INHPC = 0.5
# Weight G to PC
w_GPC = 0.9
# Weigth S to PC
w_SPC = 0.8

######################
el = -68
ee = -60
vth_e = -36
tr_e = 8 * ms
r_max = 20 * Hz
ie = 65
lambda_PL = 0.15 * metre

# Pyramidal Cells
eqs = '''
dv/dt = (I -(v-el) +ie)/tau : 1
x : metre
y : metre
tau : second
I : 1
'''




PC = NeuronGroup(rows * cols, eqs, threshold='v>vth_e', reset='v = ee', refractory=tr_e, method='euler')

# initialize the grid positions
grid_dist = 4 * meter
PC.tau = '50*ms'
PC.x = '(i // rows) * grid_dist'
PC.y = '(i % rows) * grid_dist'
PC.v = -60
#
# Will play the role of inputs:
PC.I = '0'


SPC = Synapses(PC, PC, 'w:1', on_pre= 'v_post += w')
#SPC.connect('i // rows - j // rows == 1 or i // rows - j // rows == - 1 ')

SPC.connect('(( i // rows - j // rows)**2 + ( i % cols -  j % cols)**2 )< 4')
SPC.w = '2*exp(-((x_pre-x_post)**2+(y_pre - y_post)**2)/(2*15*metre)**2)'


distances_list = []
for i in range(1, rows):
    distances_list.append(0.5*exp(-((PC.x[0]-PC.x[i])**4+(PC.y[0] - PC.y[1])**4)/(2*15*metre)**4))
plot_distrib(distances_list, "Distances to 1st neuron")


# -0.05*exp(-((x_pre-x_post)**2+(y_pre - y_post)**2)/(2*15*metre)**2)

# 'exp(-((x_pre-x_post)**2+(y_pre - y_post)**2)/(2*0.15*metre**2))
#exp(-((PC.x[i]-PC.x[j])**2+(PC.y[i] - PC.y[j])**2)/(2*15)**2)
m_ = max(PC.x)

# Inhibitory

ei = -80
vth_i = -50
tr_i = 2 * ms
gi = 1

eqs2 = '''
dv/dt = (I-(v-ei) + ie)/tau : 1
tau : second
I : 1
'''

INH = NeuronGroup(int(rows * cols / 10), eqs2, threshold='v>vth_i', reset='v = ei', refractory=tr_i, method='euler')
INH.I = 0
INH.tau = 5 * ms

SPCINH = Synapses(INH, PC, on_pre='v_post-=0.01')
SPCINH.connect(p=0.)

SINHPC = Synapses(PC, INH, on_pre='v_post+=0.01')
SINHPC.connect(p=0.)

G = NeuronGroup(rows, eqs2, threshold='v>vth_i', reset='v = ei', refractory=tr_i, method='euler')
G.I = - 10
G.tau = 5 * ms

SPCG = Synapses(G, PC, on_pre='v_post+=0.1')
SPCG.connect('((j % rows <= 17 and j%rows >= 15) and (j // cols <= 10 )) or ((j // cols >= 8 and j // cols <= 10) and (j % rows <= 15 and j%rows >= 4) or ((j//cols>= 10) and (j%rows >=4 and j%rows<=6)))')


S = NeuronGroup(1, eqs2, threshold='v>vth_i', reset='v = ei', refractory=tr_i, method='euler')
S.I = - 10
S.tau = 5 * ms

SS = Synapses(G, PC, on_pre='v_post+=0.5')
SS.connect('((j % rows <= 17 and j%rows >= 15) and (j // cols == 0 ))')
# or ((j//cols== 19) and (j%rows >=4 and j%rows<=6))')

# or ()
