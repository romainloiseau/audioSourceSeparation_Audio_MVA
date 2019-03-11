import numpy as np
from tqdm import tqdm_notebook
import utils

def update_Q(Q, W, H, V, V_hat, Kpart):
    I, F, N = V.shape
    J = W.shape[0]
    WH = np.zeros((J, F, N), dtype=complex)
    for j in range(J):
        W_masked, H_masked = utils.W_H_masked(W, H, j, Kpart)
        WH = W_masked.dot(H_masked)
        for i in range(I):
            Q[i, j] = np.multiply(Q[i, j], 
                                  np.divide(((1/(V_hat[i] * V_hat[i])) * V[i] * WH).dot(np.ones(N, dtype=complex)),
                                            ((1/V_hat[i]) * WH).dot(np.ones(N, dtype=complex))
                                           )
                                 )
    return Q

def update_W(Q, W, H, V, V_hat, Kpart):
    I, F, N = V.shape
    J, _, K = W.shape
    ind = np.cumsum(Kpart)

    for j in range(J):
        prev = ind[j - 1] if j > 0 else 0
        s1 = np.zeros((F, F), dtype=complex)
        s2 = np.zeros((F, F), dtype=complex)
        for i in range(I):
            s1 += np.diag(Q[i, j]).dot((1/(V_hat[i]*V_hat[i])) * V[i])
            s2 += np.diag(Q[i, j]).dot(1/V_hat[i])
        W[:, prev:ind[j]] = np.multiply(W[:, prev:ind[j]],
                           np.divide(s1.dot(H[prev:ind[j]].transpose()),
                                     s2.dot(H[prev:ind[j]].transpose())
                                    )
                          )
    return W

def update_H(Q, W, H, V, V_hat, Kpart):
    I, F, N = V.shape
    J, _, K = W.shape
    ind = np.cumsum(Kpart)

    for j in range(J):
        prev = ind[j - 1] if j > 0 else 0
        s1 = np.zeros((F, F), dtype=complex)
        s2 = np.zeros((F, F), dtype=complex)
        for i in range(I):
            s1 += np.diag(Q[i, j]).dot(V_hat_inv[i].dot(V_hat_inv[i]).dot(V[i]))
            s2 += np.diag(Q[i, j]).dot(V_hat_inv[i])
        H[prev:ind[j]] = np.multiply(H[prev:ind[j]],
                           np.divide(W[:, prev:ind[j]].transpose().dot(s1),
                                     W[:, prev:ind[j]].transpose().dot(s2)
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
    V_hat = compute_Vhat(Qrond, W, H)
    Q = update_Q(Q, W, H, V, V_hat, Kpart)
    W = update_W(Q, W, H, V, V_hat, Kpart)
    H = update_H(Q, W, H, V, V_hat, Kpart)
    Q, W, H = normalize(Q, W, H, Kpart)
    return Q, W, H

