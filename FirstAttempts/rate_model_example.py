import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.linalg
import scipy.spatial.distance
from pylab import savefig


data_path = "C:\\Users\\Victoria\\Desktop\\ETH\\SemesterProject\\"

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def binom(n, k):
    return math.factorial(n) // math.factorial(k) // math.factorial(n - k)

# Normalize rows of a non-negative matrix
def normRows(M):
    for row in M:
        row /= np.sum(row)
    return M

def step_linear(x, theta, beta):
    xNew = x
    xNew[x < theta] = 0.
    xNew[x > theta + 1./beta] = 1.
    xNew[(x > theta) & (x < theta + 1./beta)] = beta * (x[(x > theta) & (x < theta + 1./beta)] - theta)
    return xNew

# Update excitatory and inhibitory populations of a neuronal rate model
def update(M_EE, M_II, M_IE, M_EI, xExc, xInh, alpha, noise, theta, beta, fun):
    excInpEff = M_EE.dot(xExc) - M_IE.dot(xInh) + np.random.uniform(0, noise, xExc.shape)
    inhInpEff = M_EI.dot(xExc) - M_II.dot(xInh) + np.random.uniform(0, noise, xInh.shape)

   # xExcNew = (1 - alpha) * xExc + alpha * fun(excInpEff, theta, beta)
   # xInhNew = (1 - alpha) * xInh + alpha * fun(inhInpEff, theta, beta)
    xExcNew = (1 - alpha) * xExc + alpha * excInpEff
    xInhNew = (1 - alpha) * xInh + alpha * inhInpEff

    return xExcNew, xInhNew

########################
#  Initialization
########################

nExc = 400   # Number of excitatory neurons
nExc1 = 200
nExc2 = 200
nInh1 = 50
nInh2 = 50
nInh = 100   # Number of inhibitory neurons
alpha = 0.1  # Neuronal leak rate (alpha = dt / tau)
noise = 0.03  # Input noise magnitude
eps1 = 0.3
eps2 = 0.8
# Neuronal population firing rates
xExc0 = np.random.uniform(0, 1, nExc)
xInh0 = np.random.uniform(0, 1, nInh)

# Synaptic weights
# Model 0
def matrices_fully_connected( nExc, nInh):
    M_EE = normRows(np.random.uniform(0, 1, (nExc, nExc)))
    M_II = normRows(np.random.uniform(0, 1, (nInh, nInh)))
    M_IE = normRows(np.random.uniform(0, 1, (nExc, nInh)))
    M_EI = normRows(np.random.uniform(0, 1, (nInh, nExc)))
    return M_EE, M_II, M_IE, M_EI

# Subpopulations
def matrices_subpopulations(nExc1, nExc2, nInh1, nInh2):

    # M_EE
    M_E1E1 = normRows(np.random.uniform(0, 1, (nExc1, nExc1)))
    M_E2E2 = normRows(np.random.uniform(0, 1, (nExc2, nExc2)))
    M_EE = scipy.linalg.block_diag(M_E1E1,M_E2E2)

    # M_II
    M_I1I1 = normRows(np.random.uniform(0, 1, (nInh1, nInh1)))
    M_I2I2 = normRows(np.random.uniform(0, 1, (nInh2, nInh2)))
    M_II = scipy.linalg.block_diag(M_I1I1,M_I2I2)

    # M_IE
    M_I1E1 = normRows(np.random.uniform(0, 1, (nExc1, nInh1)))
    M_I2E2 = normRows(np.random.uniform(0, 1, (nExc2, nInh2)))
    M_IE = scipy.linalg.block_diag(M_I1E1,M_I2E2)

    # M_EI
    M_E1I2 = normRows(np.random.uniform(0, 1, (nInh2, nExc1)))
    M_E2I1 = normRows(np.random.uniform(0, 1, (nInh1, nExc2)))
    M_EI = scipy.linalg.block_diag(M_E1I2,M_E2I1)

    return M_EE, M_II, M_IE, M_EI

M_EE, M_II, M_IE, M_EI =  matrices_subpopulations(nExc1, nExc2, nInh1, nInh2)
#M_EE, M_II, M_IE, M_EI =  matrices_fully_connected(nExc1, nExc2, nInh1, nInh2)

# Exitatory and Inhibatory power proportional to distance
# Distance
def dist_matrix(num):
    k = binom(num, 2)
    v = np.random.randint(0, nExc, k)
    dist_matrix = scipy.spatial.distance.squareform(v)
    return dist_matrix

def M_from_dist(n, dist_matrix,  theta, eps1, eps2):
    M = np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, n):
            if dist_matrix[i,j] < theta:
                M[i,j] = np.random.uniform(eps2, 1)
            else:
                M[i, j] = np.random.uniform(0, eps1)
    return normRows(M)

def matrices_distance(nExc, nInh, theta, eps1 = 0.3, eps2 = 0.8):
    # M_EE = normRows(np.random.uniform(0, 1, (nExc, nExc)))
    M_EE = M_from_dist(nExc, dist_matrix(nExc), theta, eps1, eps2)
    M_II = M_from_dist(nInh, dist_matrix(nInh), theta, eps1, eps2)
    # M_II = normRows(np.random.uniform(0, 1, (nInh, nInh)))
    M_EI = normRows(np.random.uniform(0, 1, (nInh, nExc)))
    M_IE = normRows(np.random.uniform(0, 1, (nExc, nInh)))
    return M_EE, M_II, M_IE, M_EI


#M_EE, M_II, M_IE, M_EI =  matrices_distance(nExc, nInh, 100)


########################
#  Analysis
########################
def spectral_analysis( M_EE, M_II, M_IE, M_EI , ALPHA, Beeta = np.array([1])):
    rho = []
    for alpha in ALPHA:
        for beta in Beeta:
            M1 = (1 - alpha) * np.eye(nExc, nExc) + alpha * beta * M_EE
            M2 = - beta * alpha * M_IE
            M3 = beta * alpha * M_EI
            M4 = (1 - alpha) * np.eye(nInh, nInh) - beta * alpha * M_II
            bmatr = np.bmat([[M1,M2],[M3,M4]])
            eig_values, eig_vectors = np.linalg.eig(bmatr)
            rho.append(np.amax(np.abs(eig_values)))
    print("rho = ")
    print(rho)
    return rho

M_EE, M_II, M_IE, M_EI = matrices_fully_connected(nExc, nInh)

#M_EE, M_II, M_IE, M_EI =  matrices_subpopulations(300, 100, nInh1, nInh2)

#M_EE, M_II, M_IE, M_EI =  matrices_distance(nExc, nInh, 100)

# When f = step_linear(alpha, beta)
Beeta = np.array([1])
ALPHA = np.array([0.01, 0.05, 0.1, 0.15, 0.2])
rho = spectral_analysis( M_EE, M_II, M_IE, M_EI, ALPHA, Beeta)


########################
#  Run Simulation
########################

def simulate(xExc, xInh, M_EE, M_II, M_IE, M_EI,  theta, beta, alpha, tSteps = 3000):


    #theta = np.array([0.02])
#beta = np.array([100])
    # We will track the average magnitudes of excitatory and inhibitory populations
    excMag = np.zeros(tSteps)
    inhMag = np.zeros(tSteps)
    excMag[0] = np.linalg.norm(xExc)
    inhMag[0] = np.linalg.norm(xInh)
    oneNeuron = np.zeros(tSteps)
    oneNeuron[0] = xExc[0]

    for t in range(1, tSteps):
        xExc, xInh = update(M_EE, M_II, M_IE, M_EI, xExc, xInh, alpha, noise, theta, beta, step_linear)

        oneNeuron[t] = np.max(xExc)
        excMag[t] = np.linalg.norm(xExc)
        inhMag[t] = np.linalg.norm(xInh)

    ########################
    #  Plotting
    ########################
    plt.semilogy(excMag, label='exc')
    plt.semilogy(inhMag, label='inh')
    plt.legend()
    #plt.title('For distance matrix for $\\epsilon_1$ =' + str(eps1) + 'and $\\epsilon_2$ = ' + str(eps2) + '$ and \\theta$ =' + str(theta) + ' and $\\beta$ = ' + str(beta))
    plt.title('For alpha ' + str(alpha))
    savefig(data_path + '\\MyTexts\\' + 'For alpha ' + str(np.int(100*alpha)) + '.png', dpi=100)
    #
    #plt.title('E1 = '+ str(nExc1) + 'and E2 = ' + str(nExc2) + ' and I1 = ' +str(nInh1) + ' and I2 = ' + str(nInh2) + ' for $\\theta$ =' + str(th) + ' and $\\beta$ = ' + str(b))
    #plt.title('For g(t) = id ')
    #savefig(data_path + '\\MyTexts\\' + 'Subpopulations_E1_300_E2_100_I1_50_I2_50theta' + str(np.int(1000 * th)) + 'beta' + str(b) + '.png', dpi=100)
    #
    #savefig(data_path + '\\MyTexts\\' + 'eps1_' +str(np.int(10 * eps1)) + 'and_eps2_' + str(np.int(10 * eps2)) + 'theta' + str(np.int(1000 * th)) + 'beta' + str(b) + '.png', dpi=100)
    #
    #savefig(data_path + '\\MyTexts\\' +'theta' + str(np.int(1000 * th)) + 'beta' + str(b) + '.png', dpi=100)
    #savefig(data_path + '\\MyTexts\\' + 'g(t)=id.png', dpi=100)

    plt.show()
    return xExc, xInh, excMag, inhMag

tSteps = 3000
# theta = np.array([0.001, 0.01, 0.02, 0.03])
theta = np.array([0])
beta = np.array([100])

def analyse_PSD(data1, data2, freq, title, label1, label2):
    freqs = np.fft.fftfreq(data1.size, freq)
    idx = np.argsort(freqs)
    ps = np.abs(np.fft.fft(data1))**2
    plt.semilogy(freqs[idx], ps[idx], label = label1)

    freqs2 = np.fft.fftfreq(data2.size, freq)
    idx2 = np.argsort(freqs2)
    ps2 = np.abs(np.fft.fft(data2))**2
    plt.semilogy(freqs2[idx2], ps2[idx2], label = label2)
    plt.title(title)

    savefig(data_path + '\\MyTexts\\' + title + '.png', dpi=100)
    plt.show()
    return freqs2, idx, ps

excMag = []
inhMag = []
freq = 1/500
for alpha in ALPHA:
    for th in theta:
        for b in beta:
            xExc, xInh, excMag_th_b, inhMag_th_b = simulate(xExc0, xInh0, M_EE, M_II, M_IE, M_EI,  th, b, alpha, tSteps )
            excMag.append(excMag_th_b)
            inhMag.append(inhMag_th_b)

            #analyse_PSD(excMag_th_b, inhMag_th_b, freq, 'FFT for theta = ' + str(th) + 'and beta = ' + str(b), 'excMag', 'inhMag',)

def plot_used_functions():
    x = (1/400) * np.arange(0, 400)
    s=[sigmoid(12*y*y*y + 45 * y- 12) for y in x]
    plt.plot(step_linear(x, 0.3, 50), label = 'step linear')
    plt.plot(np.tanh(x), label = 'tanh function')

    plt.plot(s, label = 'sigmoid s')
    plt.legend()
    plt.show()
    return 0
#plot_used_functions()