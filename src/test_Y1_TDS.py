
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


def test_var_non_negativity():
    assert sigma_11 >= 0
    
def test_covar_sqr_matrix():
    r, c = Omega_1.shape
    assert r == c

def test_number_of_G_estimates():
    p = G_test.shape[1]
    assert len(b1) == p

def test_number_of_Z_estimates():
    l = Z_test.shape[1]
    assert len(g1) == l
    
def test_sum_q():
    assert_almost_equal(q.sum(), 1)
    
def test_unique_pairs_known1():
    # should have 50 unique pairs
    g = npr.binomial(2, 0.4, (50,1))
    z = npr.normal(size=(50,1))
    ind, g_u, z_u = tds.unique_pairs(g,z)
    r, c = ind.shape
    assert r == c
    
def test_unique_pairs_known2():
    # if g \in {0,1,2} and z \in {0,1}, then we should have at most 6 unique pairs
    g = npr.binomial(2, 0.4, (50,1))
    z = npr.binomial(1, 0.45, (50,1))
    ind, g_u, z_u = tds.unique_pairs(g,z)
    assert (g_u.shape[0] == z_u.shape[0]) <= 6
    