import numpy as np
import numpy.random as npr
import src.QT_analysis_functions as tds

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


def test_var_non_negativity():
    assert sigma_11 >= 0
    
def test_covar_sqr_matrix():
    r, c = Omega_1.shape
    assert r == c