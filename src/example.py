
import numpy as np
import numpy.random as npr
import src.QT_analysis_functions as tds
npr.seed(123451)
n_test = 250
n1_test = 100
m_test = n1_test

Y_test = npr.normal(loc=3,scale=1,size=(n_test,1))
G_test = npr.binomial(2, 0.45, (n1_test,1))
Z_test = npr.normal(size=(n1_test,1))

beta = np.array([0.0])
gamma = np.array([0.0])
s_11 = np.var(Y_test)
q_j = np.repeat(np.array([1./m_test]),n1_test)

n2_test = 50
Y2_test = npr.normal(size=(n2_test,1))
G2_test = npr.binomial(2, 0.4, (n2_test,1))
Z2_test = npr.normal(size=(n2_test,1))

res = tds.MLE_TDS(Y_test, Y2_test, G_test, G2_test, Z_test, Z2_test, beta, gamma, s_11, q_j)
print res.y1
print res.y2
# print res.covar_y1 (large array)
# print res.covar_y2