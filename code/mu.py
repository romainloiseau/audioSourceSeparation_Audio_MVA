import numpy as np
from tqdm import tqdm_notebook
import utils

def update_Q(Q, W, H, V, V_hat_inv):
    I, F, N = V.shape
    J = W.shape[0]
    WH = np.zeros((J, F, N), dtype=complex)
    for j in range(J):
        WH = W[j].dot(H[j])
        for i in range(I):
            Q[i, j] = np.multiply(Q[i, j], 
                                  np.divide(V_hat_inv[i].dot(V_hat_inv[i]).dot(V[i]).dot(WH).dot(np.ones(N, dtype=complex)),
                                            V_hat_inv[i].dot(WH).dot(np.ones(N, dtype=complex))
                                           )
                                 )
    return Q

def update_W(Q, W, H, V, V_hat_inv):
    I, F, N = V.shape
    J, _, K = W.shape
    for j in range(J):
        s1 = np.zeros((F, F), dtype=complex)
        s2 = np.zeros((F, F), dtype=complex)
        for i in range(I):
            s1 += np.diag(Q[i, j]).dot(V_hat_inv[i].dot(V_hat_inv[i]).dot(V[i]))
            s2 += np.diag(Q[i, j]).dot(V_hat_inv[i])
        W[j] = np.multiply(W[j],
                           np.divide(s1.dot(H[j].transpose()),
                                     s2.dot(H[j].transpose())
                                    )
                          )
    return W

def update_H(Q, W, H, V, V_hat_inv):
    I, F, N = V.shape
    J, _, K = W.shape
    
    for j in range(J):
        s1 = np.zeros((F, F), dtype=complex)
        s2 = np.zeros((F, F), dtype=complex)
        for i in range(I):
            s1 += np.diag(Q[i, j]).dot(V_hat_inv[i].dot(V_hat_inv[i]).dot(V[i]))
            s2 += np.diag(Q[i, j]).dot(V_hat_inv[i])
        H[j] = np.multiply(H[j],
                           np.divide(W[j].transpose().dot(s1),
                                     W[j].transpose().dot(s2)
                                    )
                          )
    return H


def normalize(Q, W, H, Kpart):
    Q = Q.transpose(1, 0, 2)
    sumQ = np.sqrt(np.sum(np.real(Q * np.conj(Q)), axis = 0))
    Q = (Q / sumQ).transpose(1, 0, 2)
    
    sumQ_W = np.array([[sumQ[:, i] for _ in range(k)] for i, k in enumerate(Kpart)]).reshape((np.sum(Kpart), -1)).transpose()
    W *= sumQ_W
    
    sumW = np.sum(np.real(W), axis = 0)
    W /= sumW
    H = (H.transpose() * sumW).transpose()
    return Q, W, H


def compute_Qrond(Q, K, Kpart):
    F, I, J = Q.shape
    ind = np.cumsum(Kpart)
    Qrond = np.zeros((F, I, K), dtype = complex)
    for j, k in enumerate(Kpart):
        cur = ind[j]
        prev = ind[j - 1] if j > 0 else 0
        Qrond[:, :, prev:cur] = np.repeat(Q[:, :, j].reshape(F, I, 1), cur - prev, axis = -1)
    return Qrond

def compute_Vhat(Qrond, W, H):
    I, K, F = Qrond.shape
    N = H.shape[1]
    V_hat = np.zeros((I, F, N), dtype=complex)
    for i in range(I):
        V_hat[i] = np.multiply(Qrond[i].transpose(), W).dot(H)
    return V_hat

def mu_iteration(Q, W, H, V, Kpart):
    F, I, J = Q.shape
    K, N = H.shape
    Qrond = compute_Qrond(Q, K, Kpart)
    V_hat = compute_Vhat(Qrond, K, Kpart)
    Q = update_Q(Q, W, H, V, V_hat_inv)
    W = update_W(Q, W, H, V, V_hat_inv)
    H = update_H(Q, W, H, V, V_hat_inv)
    Q, W, H = normalize(Q, W, H, Kpart)
    return Q, W, H
    