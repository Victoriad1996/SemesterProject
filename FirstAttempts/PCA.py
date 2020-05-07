from numpy import loadtxt
from pylab import figure, plot, xlabel, grid, legend, title, savefig
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


#Here we take the values from the loaded data for the positions.
n=5
file_name='two_springs.dat'
plot_name='Mass positions for the Coupled Mass System'
# t, x1, xy, x2, y2, x3, y3 = loadtxt('two_springs.dat', unpack=True)
t = loadtxt(file_name, unpack=True)[0]
z = []
w = []
for i in range(n):
    z.append(loadtxt(file_name, unpack=True)[2 * i + 1])
    # w.append(loadtxt('two_springs.dat', unpack=True)[2*(i+1)])
# z.append(loadtxt('two_springs.dat', unpack=True)[2*i+1])
figure(1, figsize=(6, 4.5))
#
xlabel('t')
grid(True)
lw = 1
#
for i in range(len(z)):
    plot(t, z[i], linewidth=lw)

title(plot_name)
#
plt.show()


#####################

#For each pair of masses we compute the principal components of the matrix (z[i],z[j)).
#And plot it along with the scatter matrix.
f='PCA.dat'
pca_list=[]
for i in range(n):
    pca_list_i=[]
    for j in range(i+1,n):

        pca=PCA(n_components=2)
        df_pca=pca.fit([z[i],z[j]])
        m=np.asmatrix([z[i],z[j]])
        pca.fit(np.transpose(m))

        comp=np.matmul(pca.components_,np.diag(pca.explained_variance_))
        pca_list_i.append(comp)
        mean=pca.mean_
        plt.scatter(z[i],z[j])
        ax = plt.axes()
        ax.arrow(mean[0], mean[1], comp[0][0], comp[1][0], head_width=0.05, head_length=0.1, fc='k', ec='k')
        ax.arrow(mean[0], mean[1], comp[0][1], comp[1][1], head_width=0.05, head_length=0.1, fc='k', ec='k')

        plt.plot(range(5))
        plt.xlim(mean[0]-3, mean[0]+3)
        plt.ylim(mean[1]-3, mean[1]+3)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.draw()
        savefig('ScatterMatrixPCA.png', dpi=100)
        title('Scatter Matrix and PCA for masses '+ str(i)+' and '+ str(j))
        savefig('Scatter Matrix and PCA for masses '+ str(i)+' and '+ str(j)+'.png')
        plt.show()
    pca_list.append(pca_list_i)

print(pca_list)
pca_list_frame=pd.DataFrame(pca_list)

#Here to download the data
# data_path="C:\\Users\\Victoria\\Desktop\\ETH\\SemesterProject\\"
# pca_list_frame.to_csv(data_path+'pca.csv')