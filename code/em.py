import numpy as np
from tqdm import tqdm_notebook
import sklearn.decomposition as skd
import utils

def compute_sigma_s(W, H, F, N, J, Kpart):
    sigma_s = np.zeros((F, N, J, J), dtype = complex)
    for j in range(J):
        W_masked, H_masked = utils.W_H_masked(W, H, j, Kpart)
        sigma_s[:, :, j, j] = W_masked.dot(H_masked)
    return sigma_s

def compute_sigma_c(W, H, F, N, K):
    sigma_c = np.zeros((F, N, K, K), dtype = complex)
    for k in range(K):
        sigma_c[:, :, k, k] = W[:, k].reshape((F, 1)).dot(H[k].reshape((1, N)))
    return sigma_c                                                   
    
def compute_Arond(A, K, Kpart):
    F, I, J = A.shape
    ind = np.cumsum(Kpart)
    Arond = np.zeros((F, I, K), dtype = complex)
    for j, k in enumerate(Kpart):
        cur = ind[j]
        prev = ind[j - 1] if j > 0 else 0
        Arond[:, :, prev:cur] = np.repeat(A[:, :, j].reshape(F, I, 1), cur - prev, axis = -1)
    return Arond

def compute_sigma_x_fn(a_f, sigma_s_fn, sigma_b_f):
    return a_f.dot(sigma_s_fn.dot(np.matrix(a_f).getH())) + sigma_b_f

def compute_gs_fn(sigma_s_fn, sigma_x_fn, a_f):
    # TODO: computation trick in the overdetermined case
    sig_x_inv = np.linalg.inv(sigma_x_fn)
    return sigma_s_fn.dot(np.matrix(a_f).getH().dot(sig_x_inv))
    
def compute_gc_fn(sigma_c_fn, sigma_x_fn, arond_f):
    sig_x_inv = np.linalg.inv(sigma_x_fn)
    return sigma_c_fn.dot(np.matrix(arond_f).getH().dot(sig_x_inv))
    
def r_hat(x1, x2 = None):
    if x2 is None:x2 = x1
    return np.mean([np.expand_dims(x1[i], axis = -1).dot(np.matrix(np.expand_dims(x2[i], axis = -1)).getH()) for i in range(x1.shape[0])], axis = 0)

def squared_module(arr):
    return np.multiply(arr, arr.conjugate()).real

def init_params(X, S, Kpart, nmf_noise = 1e-2):
    F, N, J = S.shape
    I = X.shape[2]
    W = []
    H = []
    for j in range(J):
        model = skd.NMF(n_components=Kpart[j], init='random', random_state=0)
        s_2 = squared_module(S[:, :, j])
        W.append(model.fit_transform(s_2) + np.random.uniform(0, nmf_noise, (F, Kpart[j])))
        H.append(model.components_ + np.random.uniform(0, nmf_noise, (Kpart[j], N)))
        
    W = np.concatenate(tuple(W), axis = 1).astype(complex)
    H = np.concatenate(tuple(H), axis = 0).astype(complex)
    A = np.zeros((F, I, J), dtype = complex)
    sigma_b = np.zeros((F, I, I), dtype = complex)
    
    Rxx = np.zeros((F, I, I), dtype = complex)
    Rxs = np.zeros((F, I, J), dtype = complex)
    Rss = np.zeros((F, J, J), dtype = complex)
    
    for f in range(F):
        Rxx[f] = r_hat(X[f])
        Rxs[f] = r_hat(X[f], S[f])
        Rss[f] = r_hat(S[f])
        A[f] = Rxs[f].dot(np.linalg.inv(Rss[f]))
        sigma_b[f] = np.diagonal(Rxx[f] - A[f].dot(np.matrix(Rxs[f]).getH()) - Rxs[f].dot(np.matrix(A[f]).getH()) + A[f].dot(Rss[f].dot(np.matrix(A[f]).getH())))
    return A, W, H, sigma_b

def compute_E_step(X, A, W, H, sigma_b, Kpart, verbose = 1):
    F, I, J = A.shape
    K, N = H.shape
    
    sigma_c = compute_sigma_c(W, H, F, N, K)
    sigma_s = compute_sigma_s(W, H, F, N, J, Kpart)
    Arond = compute_Arond(A, K, Kpart)
    
    Rxx = np.zeros((F, I, I), dtype = complex)
    Rxs = np.zeros((F, I, J), dtype = complex)
    Rss = np.zeros((F, J, J), dtype = complex)
    U = np.zeros((F, N, K), dtype = complex)
    S = np.zeros((F, N, J), dtype = complex)

    for f in tqdm_notebook(range(F), leave = verbose > 1):
        c_f = np.zeros((N, K), dtype = complex)
        for n in range(N):
            sigma_x_fn = compute_sigma_x_fn(A[f], sigma_s[f, n], sigma_b[f])
            
            #Checking invertibility
            if verbose > 0:
                if np.linalg.det(sigma_x_fn) == 0:
                    print('\Sigma_s(f, n) = ')
                    print(sigma_s[f, n])
                    w = np.zeros((J, J), dtype = complex)
                    for j in range(J):
                        W_masked, H_masked = utils.W_H_masked(W, H, j, Kpart)
                        print('j = {}'.format(j))
                        print('W: {}'.format(W_masked[f]))
                        print('H: {}'.format(H_masked[:, n]))

            gs_fn = compute_gs_fn(sigma_s[f, n], sigma_x_fn, A[f])
            gc_fn = compute_gc_fn(sigma_c[f, n], sigma_x_fn, Arond[f])
            
            S[f, n] = gs_fn.dot(X[f, n])
            c_f[n] = gc_fn.dot(X[f, n])
            U[f, n] = np.diagonal(c_f[n].dot(np.matrix(c_f[n]).getH()) + sigma_c[f, n] - gc_fn.dot(Arond[f].dot(sigma_c[f, n])))
            
        Rxx[f] = r_hat(X[f])
        Rxs[f] = r_hat(X[f], S[f])
        Rss[f] = r_hat(S[f]) + sigma_s[f, n] - gs_fn.dot(A[f].dot(sigma_s[f, n]))
        
    return Rxx, Rxs, Rss, U, S

def compute_M_step(Rxx, Rxs, Rss, U, W, H):
    F, I, J = Rxs.shape
    K, N = H.shape
    
    A = np.zeros((F, I, J), dtype = complex)
    sigma_b = np.zeros((F, I, I), dtype = complex)
    
    for f in range(F):
        A[f] = Rxs[f].dot(np.linalg.inv(Rss[f]))
        sigma_b[f] = np.diagonal(Rxx[f] - A[f].dot(np.matrix(Rxs[f]).getH()) - Rxs[f].dot(np.matrix(A[f]).getH()) + A[f].dot(Rss[f].dot(np.matrix(A[f]).getH())))
        for k in range(K):
            W[f, k] = np.mean(np.divide(U[f, :, k], H[k])) #/ N
    
    for k in range(K):
        for n in range(N):
            H[k, n] = np.mean(np.divide(U[:, n, k], W[:, k])) #/F

    return A, sigma_b, W, H