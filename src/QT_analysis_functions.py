'''
Functions: Y1_loglike
           EM_Y1
           unique_pairs
           Y1_covar
           Y2_est
           Y2_covar
           MLE_TDS
           
All the functions necessary to estimate the parameters for primary and secondary trait under trait-dependent sampling.          
'''

def Y1_loglike(Y,G,Z,G_unique,Z_unique,beta,gamma,sigma_11,q_j,GZ_indicator):
    '''Evaluates the log likelihood function for the primary trait.
    Input: Y1 - primary trait (n obs)
           G, Z - genotype and covariates (n1 obs)
           G_unique, Z_unique - pairs of unique genotype and covariates among n1 obs (m pairs, can be at most n1)
           beta, sigma - coefficient estimates for genotype and covariates
           sigma_11 - variance 
           q_j - probability vectors for unique pairs of G,Z
           GZ_indicator - a matrix (n1xm) that indicates (0 or 1) which observation corresponds to which unique pair
    Output: returns evaluated log likelihood using the inputs.'''
    import numpy as np
    import scipy.stats as stats
    n = Y.shape[0]
    n1 = G.shape[0]
    m = len(q_j)
    
    # The log likelihood funtion has 2 components: the first one involves only samples that has genotypes and covariates 
    # information
    mu_i = np.dot(G,beta) + np.dot(Z,gamma)
    sum_n1 = (stats.norm.logpdf(Y[0:n1,0], mu_i, np.sqrt(sigma_11)) + np.log(np.dot(q_j,GZ_indicator))).sum()
    # the second component involves samples with no genotype
    mu_j = np.dot(G_unique,beta) + np.dot(Z_unique,gamma)
    sum_pdf = [(stats.norm.pdf(Y[k,0],mu_j,np.sqrt(sigma_11))*q_j).sum() for k in range(n1,n)]
    sum_n2 = np.log(sum_pdf).sum()
    
    return(sum_n1 + sum_n2)


def EM_Y1(Y, G, Z, b_0, g_0, sigma_11_0, q_0, tol=1e-6, max_iter=1000):
    '''Performs EM algorithm on the primary trait
    Input: Y - primary trait (n obs)
           G, Z - genotype and covariates (n1 obs)
           b_0, g_0, sigma_11_0, q_0 - initial values for the parameters to be estimated
    Output: returns the estimates for beta, gamma, sigma_11, q, and covariance matrix'''
    import scipy.stats as stats
    import numpy as np
    import scipy.linalg as la
    
    n = Y.shape[0]
    n1, p = G.shape
   
    # m unique pairs of observed values, ind_gz is a n1 x m array
    ind_gz, G_uni, Z_uni = unique_pairs(G, Z)   
    m = ind_gz.shape[1]
    
    ll_old = 0.0
    b1 = b_0
    g1 = g_0
    sigma_11 = sigma_11_0
    q = q_0
    iters = 0
    for iters in range(max_iter):
        iters += 1
        ll_new = 0.0
        
        # E-step
        psi_n2 = [] # psi matrix
        mus_m = np.dot(G_uni,b1) + np.dot(Z_uni,g1)
        for i in range(n1,n):
            numer = stats.norm.pdf(Y[i,0],mus_m,np.sqrt(sigma_11))
            denom = sum(numer)
            psi_n2 += [x/denom for x in numer]
        psi_n = np.vstack((ind_gz,np.array(psi_n2).reshape(((n-n1),m))))
        # M-step
        ### update b1 and g1
        W = np.vstack((G_uni.T,Z_uni.T))
        a11 = np.dot((G_uni**2).T,psi_n.T).sum()
        a12 = np.dot(np.multiply(G_uni,Z_uni).T,psi_n.T).sum()
        a22 = np.dot((Z**2).T,psi_n.T).sum()
        eta_hat = np.dot(la.inv(np.array([[a11,a12],[a12,a22]])),np.dot(np.dot(W,psi_n.T),Y))
        b1 = eta_hat[0,:]
        g1 = eta_hat[1,:]

        ### update sigma_11
        a1 = (np.tile(Y,(1,m)) - np.tile(np.dot(eta_hat.T,W),(n,1)))**2
        sigma_11 = (1./n)*np.dot(a1.T,psi_n).sum()

        ### update q_j
        q = (1./n)*psi_n.sum(0)

        ### compute new log likelihood value
        ll_new = Y1_loglike(Y, G, Z, G_uni, Z_uni, b1, g1, sigma_11, q, ind_gz)
        if np.abs(ll_new - ll_old) < tol:
            break

        ll_old = ll_new
    Omega_1 = Y1_covar(Y, G, Z, b1, g1, sigma_11, q, psi_n, m) # estimate covariance matrix
    return(b1,g1,sigma_11,q,Omega_1)


def unique_pairs(G, Z):
    '''Identifies unique pairs of genotype of covariate
    Input: G, Z - genotype and covariates
    Output: returns a (n1xm) indicator matrix and the unique genotype and covariate pairs'''
    from sets import Set
    from itertools import combinations, chain
    import numpy as np
    n1 = G.shape[0]
    chain_GZ = [zip(l,list(chain(*Z.tolist()))) for l in combinations(list(chain(*G.tolist())),len(Z))]
    GZ = list(chain(*chain_GZ))
    unique_gz = list(Set(GZ)) # obtain unique pairs through Set
    dict_gz = dict(zip(iter(unique_gz),range(len(unique_gz))))
    gz_value_frm_dict = [dict_gz[l] for l in GZ]

    m = len(unique_gz)

    if(m == n1):
        ind_gz = np.identity(m)      
    else:
        ind_gz = np.zeros((n1,len(unique_gz)))
        for r in range(ind_gz.shape[0]):
            ind_gz[r,gz_value_frm_dict[r]] = 1 
    
    gz_arr = np.array(unique_gz) # indicator matrix for which sample corresponds to which unique pair
    g_new = gz_arr[:,0].reshape((gz_arr.shape[0],1))
    z_new = gz_arr[:,1].reshape((gz_arr.shape[0],1))

    return ind_gz, g_new, z_new


def Y1_covar(Y, G, Z, b1, g1, sigma_11, q, psi_n, m):
    '''Estimates the covariance matrix of all the parameters for primary trait
    Input: Y - primary trait (n obs)
           G,Z - genotype and covariates
           b1, g1, sigma_11, q - parameter estimates
           psi_n - a (nxm) matrix of probalities for each sample with respect to each unique pairs
           m - number of unique pairs
    Output: returns the covariance matrix'''
    import numpy as np
    import scipy.linalg as la
    import sympy
    n = Y.shape[0]
    # symbolic equation 1st and 2nd derivatives of U
    b, g, s, q_s, y_s, G_s, Z_s = sympy.symbols('b g s q_s y_s G_s Z_s')
    U = sympy.symbols('U',cls = sympy.Function)
    U =  -0.5*sympy.log(s) - (0.5/s)*(y_s - b*G_s - g*Z_s)**2 + sympy.log(q_s)
    grad_sym = sympy.Matrix([b, g, s, q_s])
    l1 = [sympy.diff(U, sym) for sym in grad_sym]
    l2 = sympy.Matrix(sympy.hessian(U, grad_sym))
    f_l1 = sympy.lambdify((b, g, s, q_s, y_s, G_s, Z_s), l1[0:3], 'numpy') # allows for easy substitution into derivatives
    f_l2 = sympy.lambdify((b, g, s, q_s, y_s, G_s, Z_s), l2[0:3,0:3], 'numpy') # allows for easy substitution into derivatives

    fl1_qs = sympy.lambdify(q_s, l1[3], 'numpy')  
    l1_qs = np.array(fl1_qs(q)).reshape((m,1))
    fl2_qs = sympy.lambdify(q_s, l2[3,3], 'numpy')
    l2_qs = np.append(np.zeros((3,m)),np.diag(fl2_qs(q)),axis=0)
    
    # There are 4 different sums needed for the estimation of covariance matrix
    part1Q = 0.0
    part2Q = 0.0
    part4Q = 0.0
    for i in range(n):
        part3Q = 0.0
        for j in range(m):
            l2T = f_l2(b1,g1,sigma_11,q[j],Y[i,0],G[j,0],Z[j,0])
            temp1 = psi_n[i,j]*np.column_stack((np.append(l2T,np.zeros((m,3)),axis=0),l2_qs))
            part1Q += temp1
            
            llT = np.vstack((np.array(f_l1(b1,g1,sigma_11,q[j],Y[i,0],G[j,0],Z[j,0])),l1_qs))
            part2Q += psi_n[i,j]*np.dot(llT,llT.T)
            part3Q += psi_n[i,j]*llT
        part4Q += np.dot(part3Q,part3Q.T)
    Q_1 = -part1Q - part2Q + part4Q
    
    D = np.append((np.identity(m+2)),[[0,0,0]+[-1]*(m-1)],axis=0) # reflect the contraint of sum(q_j) = 1
    F = np.dot(np.dot(D.T,Q_1),D)
    Omega_1 = la.inv(F)
    return(Omega_1)


def Y2_est(Y1, G1, Z1, b1, g1, Y2, G2, Z2, Omega_bg_1):
    '''Estimates parameters and covariance matrix of secondary traits
    Input: Y1 - primary trait (n obs)
           G1,Z1 - genotype and covariates for primary trait (n1 obs)
           b1, g1 - beta and gamma estimates for primary trait
           Y2 - secondary trait (n2 < n1 obs)
           G2,Z2 - genotype and covariates for secondary trait (n2 obs)
           Omega_bg_1 - covariance matrix of primary trait
    Output: returns parameter estimates (delta, b2, g2, sigma_22) and covariance matrix Omega_2 '''
    import statsmodels.api as sm
    import numpy as np
    n2 = Y2.shape[0]
    Y1_hat = Y1 - np.dot(G1,b1) - np.dot(Z1,g1)
    X = np.column_stack((Y1_hat,G2,Z2))
    est = sm.OLS(Y2,X).fit() # OLS estimates of parameters
    delta, b2, g2 = est.params
    sigma_22 = (1./n2)*((Y2 - delta*Y1_hat - np.dot(G2,b2) - np.dot(Z2,g2))**2).sum() # obtain sigma_22
    Omega_2 = Y2_covar(delta,b2,g2,sigma_22,b1,g1,Y1,Y2,G1,G2,Z1,Z2,Omega_bg_1) # covariance matrix estimation
    
    return(delta, b2, g2, sigma_22, Omega_2)


def Y2_covar(delta, b2, g2, sigma_22, b1, g1, Y_1, Y_2, G_1, G_2, Z_1, Z_2, Omega_bg_1):
    '''Estimates the covariance matrix for secondary trait
    Input: delta,b2,g2,sigma_22 - parameters estimates of secondary trait
           b1,g1 - parameters estimates of primary trait
           Y1 - primary trait
           Y2 - secondary trait
           G1,Z1 - genotype and covariates of primary trait
           G2,Z2 - genotype and covariates of secondary trait
           Omega_bg_1 - covariance matrix of primary trait
    Output: returns covariance matrix of secondary trait'''
    import scipy.linalg as la
    import numpy as np
    Y1_hat = Y_1 - b1*G_1 - g1*Z_1
    Y1_G2 = np.dot(Y1_hat.T,G_2)
    Y1_Z2 = np.dot(Y1_hat.T,Z_2)
    G2_Z2 = np.dot(G_2.T,Z_2)
    G2_2 = np.dot(G_2.T,G_2)
    Z2_2 = np.dot(Z_2.T,Z_2)
    G1_Z1 = np.dot(G_1.T,Z_1)
    U = np.array([np.dot(Y1_hat.T,Y1_hat),Y1_G2,Y1_Z2,Y1_G2,G2_2,G2_Z2,Y1_Z2,G2_Z2,Z2_2]).reshape((3,3))
    V = np.array([np.dot(Y_2.T,Y1_hat),np.dot(Y_2.T,G_2),np.dot(Y_2.T,Z_2)]).reshape((3,1))
    dU_b1 = np.array([-2*G_1.sum()+2*b1*G2_2+2*g1*G1_Z1,-np.dot(G_1.T,G_2),-np.dot(G_1.T,Z_2),-np.dot(G_1.T,G_2),[0],[0],
                      -np.dot(G_1.T,Z_2),[0],[0]]).reshape((3,3)) 
    dV_b1 = np.array([-np.dot(G_1.T,Y_2),[0],[0]]).reshape((3,1))
    dU_g1 = np.array([-2*Z_1.sum()+2*g1*Z2_2+2*b1*G1_Z1,-np.dot(G_2.T,Z_1),-np.dot(Z_1.T,Z_2),
                      -np.dot(G_2.T,Z_1),[0],[0],-np.dot(Z_1.T,Z_2),[0],[0]]).reshape((3,3))
    dV_g1 = np.array([-np.dot(Z_1.T,Y_2),[0],[0]]).reshape((3,1))
    
    U_inv = la.inv(U)
    J = np.column_stack((-np.dot(np.dot(np.dot(U_inv,dU_b1),U_inv),V) + np.dot(U_inv,dV_b1),
              -np.dot(np.dot(np.dot(U_inv,dU_g1),U_inv),V) + np.dot(U_inv,dV_g1)))
    
    Omega_2 = sigma_22*U_inv + np.dot(np.dot(J,Omega_bg_1),J.T)
    return(Omega_2)


def MLE_TDS(Y1, Y2, G1, G2, Z1, Z2, b_0, g_0, sigma_11_0, q_0, tol=1e-6, max_iter=1000):
    from tabulate import tabulate
    import numpy as np
    
    class tds_output():
        pass
    
    res = tds_output()
    p = G1.shape[1]
    c = Z1.shape[1]
    # EM algorithm to estimate parameters of primary trait
    b1, g1, sigma_11, q, Omega_1 = EM_Y1(Y1, G1, Z1, b_0, g_0, sigma_11_0, q_0, tol, max_iter)
    tab1 = [['beta_1',b1],['gamma_1',g1],['sigma_11',sigma_11]]
    res.y1 = tabulate(tab1)
    n2 = G2.shape[0]
    Y1_2 = Y1[0:n2,0].reshape((n2,1))
    G1_2 = G1[0:n2,0].reshape((n2,1))
    Z1_2 = Z1[0:n2,0].reshape((n2,1))
    # OLS to estimate parameters of secondary trait
    delta, b2, g2, sigma_22, Omega_2 = Y2_est(Y1_2, G1_2, Z1_2, b1.reshape((1,1)), g1.reshape((1,1)), 
                                              Y2, G2, Z2, Omega_1[0:(p+c),0:(p+c)])
    tab2 = [['delta',delta],['beta_2',b2],['gamma_2',g2],['sigma_22',sigma_22]]
    res.y2 = tabulate(tab2)
    res.covar_y1 = Omega_1 # covariance matrix of primary
    res.covar_y2 = Omega_2 # covariance matrix of secondary
    
    return res
