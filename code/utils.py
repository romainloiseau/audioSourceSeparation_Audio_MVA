import numpy as np
import matplotlib.pyplot as plt

def plot_stft(Z):
    
    plt.figure(figsize = (15, 5))
    eps = 1e-10
    for i in range(Z.shape[0]):
        plt.subplot(101 + Z.shape[0] * 10 + i)

        logmag = np.flipud(np.log(eps + np.real(Z[i])**2))
        plt.imshow(logmag, extent=[0, logmag.shape[-1], 0, 512], aspect = 'auto', cmap = plt.cm.gist_heat)
        plt.title("STFT Spectrogram for channel {}".format(i))
        plt.ylabel("Frequencies")
        plt.xlabel("Time")
    plt.show()