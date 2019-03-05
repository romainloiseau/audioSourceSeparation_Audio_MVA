import numpy as np
from tqdm import tqdm_notebook


def compute_Sc(W, H):
    F, K = W.shape
    _, N = H.shape
    
    W = np.array([W for n in range(N)]).transpose(1, 0, 2)
    H = np.array([H for f in range(F)]).transpose(0, 2, 1)
    return W * H


def compute_Arond(A, K, Kcal):
    I, _, F = A.shape
    Arond = np.zeros((I, K, F), dtype=complex)
    for j, k in enumerate(Kcal):
        Arond[:, k] = A[:, j]
    return Arond


def compute_E_step(x_four, A, W, H, Sb, Kcal):
    Sc = compute_Sc(W, H)
    
    F, N, K = Sc.shape
    I, J, _ = A.shape
    
    Ss = Sc[:, :, Kcal]
    Arond = compute_Arond(A, K, Kcal)
    # TODO: how are we supposed to initialize the parameters ?
    # (EM method only guarantees convergence to a critical point)
    Gs = np.empty((F, N, J, I), dtype=complex)
    Gc = np.empty((F, N, K, I), dtype=complex)
    s = np.empty((F, N, J), dtype=complex)
    c = np.empty((F, N, K), dtype=complex)
    
    Rxs = np.empty((F, I, J), dtype=complex)
    Rss = np.empty((F, J, J), dtype=complex)
    u = np.empty((K, F, N), dtype=complex)
    
    for f in tqdm_notebook(range(F)):
        for n in range(N):
            a = np.matrix(A[:, :, f])
            arond = np.matrix(Arond[:, :, f])
            ss = np.diag(Ss[f, n])
            sc = np.diag(Sc[f, n])
            
            sx = a.dot(ss).dot(a.getH()) + Sb[f]
            invsx = np.linalg.inv(sx)
            
            # TODO: woodbury matrix trick in the overdetermined case
            Gs[f, n] = ss.dot(a.getH()).dot(invsx)
            
            Gc[f, n] = sc.dot(arond.getH()).dot(invsx)
            
            s[f, n] = Gs[f, n].dot(x_four[:, f, n])
            c[f, n] = Gc[f, n].dot(x_four[:, f, n])
            
            Rxs[f] += np.matrix(x_four[:, f, n]).transpose().dot(np.matrix(s[f, n]).getH().transpose()) / N
            
            Rss[f] += np.matrix(s[f, n]).transpose().dot(np.matrix(s[f, n]).getH().transpose()) / N
            Rss[f] += ss
            Rss[f] -= Gs[f, n].dot(a).dot(ss)
            
            u[:, f, n] = np.matrix(c[f, n]) * np.matrix(c[f, n]).getH() + \
                         np.diagonal(sc + Gc[f, n].dot(arond).dot(sc))
            
    return Rxs, Rss, u

    
def compute_M_step(Rxx, Rxs, Rss, u, H, W):
    F, I, J = Rxs.shape
    K, N = H.shape
    
    A = np.empty((I, J, F), dtype=complex)
    Sb = np.empty((F, I, I), dtype=complex)

    for f in range(F):

        A[:, :, f] = Rxs[f].dot(np.linalg.inv(Rss[f]))

        af = np.matrix(A[:, :, f])
        rxsf = np.matrix(Rxs[f])
        
        Sb[f] = - np.array(af.dot(rxsf.getH())) - np.array(rxsf.dot(af.getH())) \
                + np.array(af.dot(Rss[f]).dot(af.getH()))
        Sb[f] += np.diag(Rxx[f, f])
        
        W[f] = np.sum(u[:, f] / H, axis = -1) / N
        
    for n in range(N):
        H[:, n] = np.sum(u[:, :, n] / W.transpose(), axis = -1) / F
            
    return A, Sb, H, W
