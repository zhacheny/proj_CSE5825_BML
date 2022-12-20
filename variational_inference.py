import numpy as np
import pandas as pd
import scipy
import scipy.stats
import matplotlib.pyplot as plt
from scipy.special import digamma
import seaborn as sns

def calculateqmu(data):
    for k in range(K):
        q_mu = np.dot(np.dot(phi[:, k], data),mean[k]) - (1/20 + sum(phi[:, k])/2) * np.dot(mean[k],mean[k])
    return q_mu

df = []
with open('hw1_1000.txt', 'r') as f:
    for line in f.readlines():
        x1, x2 = line.strip().split()
        df.append([np.float(x1), np.float(x2)])
df = np.array(df)

# initialize hyperparameters
K = 3
n = df.shape[0]
# initialize all parameters.
sigma = np.array([np.eye(2)] * 3)
phi = scipy.stats.dirichlet.rvs([1, 1, 1], size=n)
mean = scipy.stats.multivariate_normal.rvs(mean=[0, 0], cov=np.eye(2), size=3)
# record the initial state
history = {'phi':[phi], 'mean':[mean], 'sigma':[sigma], 'qmu':[calculateqmu(df)]}


def VI(data, iterations):

    # start updating
    for iteration in range(iterations):
        print('Current Iteration:' + str(iteration))
        # update for phi
        for i in range(n):
            phi_new = []
            for k in range(K):
                update_phi = np.exp(np.dot(data[i], mean[k]) - 0.5 * (np.dot(mean[k],mean[k])))
                phi_new.append(update_phi)
            phi_new = np.array(phi_new) / sum(phi_new)
            phi[i] = phi_new
        # update for mean and sigma
        for k in range(K):
            mean[k] = np.dot(phi[:, k], data) / (0.1 + sum(phi[:, k]))
            sigma[k] = np.array([[1 / (0.1 + sum(phi[:, k])), 0], [0, 1 / (0.1 + sum(phi[:, k]))]])
        #record the parameters for each updating process
        history['phi'].append(phi.copy())
        history['mean'].append(mean.copy())
        history['sigma'].append(sigma.copy())
        history['qmu'].append(calculateqmu(data))

# start updating
print("initial state of phi:" + str(sum(phi)))
iterations = 100
VI(df, iterations)
print("phi after upadated:" + str(sum(phi)))
df_plot = pd.DataFrame({'x':df[:,0], 'y':df[:,1], 'label':np.argmax(phi, axis=-1)})
sns.set()
plt.title('ELBO of mu')
plt.plot(np.arange(len(history['qmu'])), history['qmu'])
plt.show()

plt.title('clustering plot')
sns.scatterplot(data=df_plot, x='x', y='y', hue='label', palette="tab10")
plt.scatter(mean[:,0],mean[:,1], color='red')
plt.show()
print('estimated mean', mean)