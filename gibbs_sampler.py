from scipy import stats
import scipy
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import seaborn as sns

# def lggamma(m):
#     return [sum(np.log(i) for i in range(1, m))]

def log_likelihood(data):
    N = sum(z_onehot)
    denominator = 0
    mu_sum = 0
    x_sum = 0
    # calculate the integrate of pi
    for k in range(0,K):
        denominator += math.lgamma(N[k]+1)
        mu_sum += np.dot(mu[k], mu[k])
    for i in range(0,n):
        x_sum += np.dot(data[i]-mu[z[i]], data[i]-mu[z[i]])
    res = (math.lgamma(n+3) - denominator) - 1/20 * mu_sum-0.5 * x_sum
    return res

# read data from txt

df = []
with open('hw1_250.txt', 'r') as f:
    for line in f.readlines():
        x, y = line.strip().split()
        df.append([np.float(x), np.float(y)])
df = np.array(df)

# initialize hyperparameters
n = df.shape[0]
K = 3

# initialize other parameters
iterations = 50
pi = scipy.stats.dirichlet.rvs([1,1,1]).reshape(-1)
z_onehot = scipy.stats.multinomial.rvs(n=1,p=pi, size=n)
temp_z = []
z = []
for per_z in z_onehot:
    temp_z=[np.where(per_z == 1)]
    z.append(temp_z)
z = np.array(z).reshape(-1)
mu = scipy.stats.multivariate_normal.rvs(mean=[0,0], cov=10*np.eye(2), size=3)
plot_data = []
# record the initial state
history = {'mu':[mu], 'z_onehot':[z_onehot], 'z':[z], 'loglikelihood':[log_likelihood(df)]}


def gibbs_sampler(data, iterations):
    #start updating
    for iteration in range(iterations):
        print('Current Iteration:' + str(iteration))
        N_sum = sum(z_onehot)
        # update mu
        for k in range(0, K):
            x_sum = sum(data[z == k])
            mean = 1/(0.1+N_sum[k]) * x_sum
            cov = np.array([[1/(0.1+N_sum[k]), 0], [0, 1/(0.1+N_sum[k])]])
            # using mean and covariance to sample a new mu
            mu_updated = scipy.stats.multivariate_normal.rvs(mean=mean, cov=cov, size=1)
            mu[k] = mu_updated
        # update z and its onehot
        for i in range(0, n):
            N_perz = sum(z_onehot)
            prob_z = []
            for k in range(0, K):
                prob_z_per_k = 1/N_perz[k] * np.exp(-0.5*np.dot(data[i]-mu[k],data[i]-mu[k]))
                prob_z.append(prob_z_per_k)
            prob_z_sumallk = sum(prob_z)
            prob_z_all = np.array(prob_z) / prob_z_sumallk
            # print(prob)
            z_onehot_updated = scipy.stats.multinomial.rvs(n=1, p=prob_z_all, size=1)
            # z_onehot_updated = np.random.multinomial(n=1,p=prob, size=1)
            # return the position of zi == k
            z_updated = np.where(z_onehot_updated.reshape(-1))[0][0]
            z_onehot[i] = z_onehot_updated
            z[i] = z_updated
        #record the parameters for each updating process
        history['mu'].append(mu.copy())
        history['z'].append(z.copy())
        history['z_onehot'].append(z_onehot.copy())
        history['loglikelihood'].append(log_likelihood(data))

# Start sampling
print("initial state of z:" + str(sum(z_onehot)))
gibbs_sampler(df, iterations)
print("z after upadated:" + str(sum(z_onehot)))

# ploting the charts
df_plot = pd.DataFrame({'x':df[:,0], 'y':df[:,1], 'label':z})
sns.set()
sns.scatterplot(data=df_plot, x='x', y='y', hue='label', palette="tab10")
plt.show()

# plot_x = []
# plot_y = []
# plot_color = []
# for one_z,seperate_data in zip(z,df):
#     plot_x.append(seperate_data[0])
#     plot_y.append(seperate_data[1])
#     if one_z == 0:
#         plot_color.append('red')
#     if one_z == 1:
#         plot_color.append('blue')
#     if one_z == 2:
#         plot_color.append('magenta')
#
# plt.title('clusters plot')
# for i in range(len(plot_x)):
#     # plotting the corresponding x with y
#     # and respective color
#     plt.scatter(plot_x[i], plot_y[i], c=plot_color[i], s=10,
#                 linewidth=0)
plt.show()

plt.title('the logprob of joint distribution')
plt.plot(np.arange(len(history['loglikelihood'])), history['loglikelihood'])
plt.show()
