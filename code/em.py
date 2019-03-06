import numpy as np
from tqdm import tqdm_notebook
import sklearn.decomposition as skd

def W_H_masked(W, H, j, Kpart):
    ind = np.cumsum(Kpart)
    prev = 0
    if j > 0:
        prev = ind[j-1]
    return W[:, prev:ind[j]], H[prev:ind[j]]


def compute_sigma_s(W, H, F, N, J, Kpart):
    sigma_s = np.zeros((F, N, J, J), dtype=np.float)
    for j in range(J):
        W_masked, H_masked = W_H_masked(W, H, j, Kpart)
        sigma_s[:, :, j, j] = W_masked.dot(H_masked)
    return sigma_s


def compute_sigma_c(W, H, F, N, K):
    sigma_c = np.zeros((F, N, K, K))
    for k in range(K):
        sigma_c[:, :, k, k] = W[:, k].reshape((F, 1)).dot(H[k].reshape((1, N)))
    return sigma_c                                                   
    
    
def compute_Arond(A, K, Kpart):
    F, I, J = A.shape
    ind = np.cumsum(Kpart)
    prev = 0
    Arond = np.zeros((F, I, K), dtype=complex)
    for j, k in enumerate(Kpart):
        cur = ind[j]
        Arond[:, :, prev:cur] = np.repeat(A[:, :, j].reshape(F, I, 1), cur-prev-1, axis=-1)        
        prev = cur
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
    if x2 is None:
        x2 = x1
    a = np.sum([np.expand_dims(x1[i], axis = -1).dot(np.matrix(np.expand_dims(x2[i], axis = -1)).getH()) for i in range(x1.shape[0])], axis = 0) / x1.shape[1]
    return a


def squared_module(arr):
    return np.multiply(arr, arr.conjugate()).real

def init_params(X, S, Kpart):
    F, N, J = S.shape
    I = X.shape[2]
    W = []
    H = []
    for j in range(J):
        model = skd.NMF(n_components=Kpart[j], init='random', random_state=0)
        s_2 = squared_module(S[:, :, j])
        W.append(model.fit_transform(s_2))
        H.append(model.components_)
    W = np.concatenate(tuple(W), axis=1)
    H = np.concatenate(tuple(H), axis=0)
    A = np.zeros((F, I, J), dtype=np.float)
    sigma_b = np.zeros((F, I, I), dtype=np.float)
    
    Rxx = np.zeros((F, I, I), dtype=complex)
    Rxs = np.zeros((F, I, J), dtype=complex)
    Rss = np.zeros((F, J, J), dtype=complex)
    
    for f in range(F):
        Rxx[f] = r_hat(X[f])
        Rxs[f] = r_hat(X[f], S[f])
        Rss[f] = r_hat(S[f])
        A[f] = Rxs[f].dot(np.linalg.inv(Rss[f]))
        sigma_b[f] = np.diagonal(Rxx[f] - A[f].dot(np.matrix(Rxs[f]).getH()) - Rxs[f].dot(np.matrix(A[f]).getH()) + A[f].dot(Rss[f].dot(np.matrix(A[f]).getH())))
    return A, W, H, sigma_b


def compute_E_step(X, A, W, H, sigma_b, Kpart):
    F, I, J = A.shape
    K, N = H.shape
    
    sigma_c = compute_sigma_c(W, H, F, N, K)
    sigma_s = compute_sigma_s(W, H, F, N, J, Kpart)
    Arond = compute_Arond(A, K, Kpart)
    
    Rxx = np.zeros((F, I, I), dtype=np.float)
    Rxs = np.zeros((F, I, J), dtype=np.float)
    Rss = np.zeros((F, J, J), dtype=np.float)
    U = np.zeros((F, N, K), dtype=np.float)
    S = np.zeros((F, N, J), dtype=np.float)

    for f in tqdm_notebook(range(F)):
        c_f = np.zeros((N, K), dtype=np.float)
        for n in range(N):
            sigma_x_fn = compute_sigma_x_fn(A[f], sigma_s[f, n], sigma_b[f])
            gs_fn = compute_gs_fn(sigma_s[f, n], sigma_x_fn, A[f])
            gc_fn = compute_gc_fn(sigma_c[f, n], sigma_x_fn, Arond[f])
            
            S[f, n] = gs_fn.dot(X[f, n])
            c_f[n] = gc_fn.dot(X[f, n])
            U[f, n] = np.diagonal(c_f[n].dot(np.matrix(c_f[n]).getH()) + sigma_c[f, n] - gc_fn.dot(Arond[f].dot(sigma_c[f, n])))
            
        Rxx[f] = r_hat(X[f])
        Rxs[f] = r_hat(X[f], S[f])
        Rss[f] = r_hat(s_f) + sigma_s[f, n] - gs_fn.dot(A[f].dot(sigma_s[f, n]))
        
    return Rxx, Rxs, Rss, U, S

    
def compute_M_step(Rxx, Rxs, Rss, U, W, H):
    F, I, J = Rxs.shape
    K, N = H.shape
    
    A = np.zeros((F, I, J), dtype=np.float)
    sigma_b = np.zeros((F, I, I), dtype=np.float)
    W = np.zeros((F, K), dtype=np.float)
    K = np.zeros((K, N), dtype=np.float)
    
    for f in range(F):
        A[f] = Rxs[f].dot(np.linalg.inv(Rss[f]))
        sigma_b[f] = np.diagonal(Rxx[f] - A[f].dot(np.matrix(Rxs[f]).getH()) - Rxs[f].dot(np.matrix(A[f]).getH()) + A[f].dot(Rss[f].dot(np.matrix(A[f]).getH())))
        for k in range(K):
            W[f, k] = (1/N) * np.sum(np.divide(U[k, f], H[k]))
    for k in range(K):
        for n in range(N):
            H[k, n] = (1/F) * np.sum(np.divide(U[k, :, n], W[:, n]))

    return A, sigma_b, W, H
