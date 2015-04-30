
import numpy as np
import numpy.random as npr
import imp
tds = imp.load_source('QT_analysis_functions', '/home/bitnami/sta-663-project/src/QT_analysis_functions.py')

# parameters setup
n = 5000
n1 = 300
n2 = 300
maf = [0.001*a for a in range(1,11)] + [0.04]
b1 = 1.5 
g1 = 0.3
b2 = 0.8
g2 = 0.1

# generate data
npr.seed(2842)
G = npr.binomial(2, 0.04, n)
Z = np.array([npr.normal(loc = 0) if G[i] == 0 else npr.normal(loc = 1) if G[i] == 1 else npr.normal(loc = 2) for i in range(n)]).reshape((n,))
error = npr.multivariate_normal(mean=[0,0],cov=[[1,0.38],[0.38,1]],size=n)
e1 = error[:,0]
e2 = error[:,1]
Y1 = b1*G + g1*Z + e1
Y2 = b2*G + g2*Z + e2
data_Y1 = np.column_stack((Y1,Y2,G,Z))
data_Y1 = data_Y1[np.argsort(data_Y1[:,0])]
data = np.vstack((data_Y1[0:(n2/2),:],data_Y1[(n-(n2/2)):n,:],data_Y1[(n2/2):(n-(n2/2)),:]))

# dataset
Y1 = data[:,0].reshape((n,1))
Y2 = data[0:n2,1].reshape((n2,1))
G = data[0:n2,2].reshape((n2,1))
Z = data[0:n2,3].reshape((n2,1))

beta = np.array([0.0])
gamma = np.array([0.0])
s_11 = np.var(Y1)
q_j = np.repeat(np.array([1./n1]),n1)
 
res = tds.MLE_TDS(Y1, Y2, G, G, Z, Z, beta, gamma, s_11, q_j)

print res.y1
print res.y2
