
import numpy as np
import numpy.random as npr
from numpy.testing import assert_almost_equal
import QT_analysis_functions as tds

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
    
b1, g1, sigma_11, q, Omega_1 = tds.EM_Y1(Y_test, G_test, Z_test, beta, gamma, s_11, q_j)

n2_test = 50
Y2_test = npr.normal(size=(n2_test,1))
G2_test = npr.binomial(2, 0.4, (n2_test,1))
Z2_test = npr.normal(size=(n2_test,1))

Y1 = np.array(Y_test[0:n2_test,0]).reshape((n2_test,1))
G1 = np.array(G_test[0:n2_test,0]).reshape((n2_test,1))
Z1 = np.array(Z_test[0:n2_test,0]).reshape((n2_test,1))
delta, b2, g2, sigma_22, Omega_2 = tds.Y2_est(Y1, G1, Z1, b1.reshape((1,1)), g1.reshape((1,1)), Y2_test, G2_test, Z2_test, Omega_1[0:2,0:2])

def test_var_non_negativity():
    assert sigma_22 >= 0
    
def test_covar_sqr_matrix():
    r, c = Omega_2.shape
    assert r == c