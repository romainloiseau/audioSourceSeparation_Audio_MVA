import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import wave

def plot_stft(freqs, Z):
    # TODO: change y axis in the plot in order to display the true frequencies (they are in the list freqs returned by the STFT)
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
    
def create_inputs(files, maxi = 1., coef_add_noise = 1e-2, coef_mult_noise = 1e-2, coef_mix = .1):
    # load source mono wave files
    rates, srcs = [], []
    for file in files:
        rate, _, src = readwav(file)
        rates.append(rate)
        srcs.append(src / maxi)
    for rate in rates[1:]:assert rate == rates[0]
    srcs = np.array(srcs)[:, :, 0]
    
    # create noised inputs by modifying source samples
    matrix = coef_mix * np.ones((srcs.shape[0], srcs.shape[0])) + np.diag((1 - srcs.shape[0] * coef_mix) * np.ones(srcs.shape[0]))
    perturbated_srcs = np.multiply(matrix.dot(srcs), np.random.normal(1, coef_mult_noise, srcs.shape)) + np.random.normal(0, coef_add_noise, srcs.shape)
    
    return perturbated_srcs, srcs
    
def resynthesize_src(S, maxi):
    times, output = sig.istft(S.transpose((2, 0, 1)))
    return output * maxi

def W_H_masked(W, H, j, Kpart):
    ind = np.cumsum(Kpart)
    prev = ind[j - 1] if j > 0 else 0
    return W[:, prev:ind[j]], H[prev:ind[j]]
    
def dIS(x, y):
    return np.sum((x / y) - np.log(x / y) - 1)

def compute_loglike(S, W, H, Kpart, epsilon = 10**(-12)):
    loglike = 0
    
    J = len(Kpart)
    S2 = np.real(S * np.conj(S))
    S2_zeros = S2 < epsilon
    
    for j in range(J):
        Wj, Hj = W_H_masked(W, H, j, Kpart)
        WjHj = np.real(Wj.dot(Hj))
        
        WjHj_zeros = WjHj < epsilon
        zeros = S2_zeros[:, :, j].astype(int) + WjHj_zeros.astype(int) > 0
        WjHj[zeros] = epsilon
        S2[:, :, j][zeros]= epsilon
        
        n_ones = 1 - np.mean(zeros)
        loglike += dIS(S2[:, :, j], WjHj) * n_ones
        
    return loglike

def plot_loglike(loglikess):
    labels = ["true", "rec"]
    for loglikes, label in zip(loglikess, labels):
        plt.plot(np.arange(len(loglikes)), loglikes, label = label)
    plt.xlabel("iterations")
    plt.ylabel("log likelihood")
    plt.show()
    
    
    
#################################################################################
#################################################################################
#################################################################################
#################################################################################
    
    
    
"""
code taken from https://gist.github.com/WarrenWeckesser/7461781
allows us to read 24-bit wav files
"""
    
    
def _wav2array(nchannels, sampwidth, data):
    """data must be the string containing the bytes from the wav file."""
    num_samples, remainder = divmod(len(data), sampwidth * nchannels)
    if remainder > 0:
        raise ValueError('The length of data is not a multiple of '
                         'sampwidth * num_channels.')
    if sampwidth > 4:
        raise ValueError("sampwidth must not be greater than 4.")

    if sampwidth == 3:
        a = np.empty((num_samples, nchannels, 4), dtype=np.uint8)
        raw_bytes = np.fromstring(data, dtype=np.uint8)
        a[:, :, :sampwidth] = raw_bytes.reshape(-1, nchannels, sampwidth)
        a[:, :, sampwidth:] = (a[:, :, sampwidth - 1:sampwidth] >> 7) * 255
        result = a.view('<i4').reshape(a.shape[:-1])
    else:
        # 8 bit samples are stored as unsigned ints; others as signed ints.
        dt_char = 'u' if sampwidth == 1 else 'i'
        a = np.fromstring(data, dtype='<%s%d' % (dt_char, sampwidth))
        result = a.reshape(-1, nchannels)
    return result


def readwav(file):
    """
    Read a wav file.
    Returns the frame rate, sample width (in bytes) and a numpy array
    containing the data.
    This function does not read compressed wav files.
    """
    wav = wave.open(file)
    rate = wav.getframerate()
    nchannels = wav.getnchannels()
    sampwidth = wav.getsampwidth()
    nframes = wav.getnframes()
    data = wav.readframes(nframes)
    wav.close()
    array = _wav2array(nchannels, sampwidth, data)
    return rate, sampwidth, array


def writewav24(filename, rate, data):
    """Create a 24 bit wav file.
    data must be "array-like", either 1- or 2-dimensional.  If it is 2-d,
    the rows are the frames (i.e. samples) and the columns are the channels.
    The data is assumed to be signed, and the values are assumed to be
    within the range of a 24 bit integer.  Floating point values are
    converted to integers.  The data is not rescaled or normalized before
    writing it to the file.
    Example: Create a 3 second 440 Hz sine wave.
    >>> rate = 22050  # samples per second
    >>> T = 3         # sample duration (seconds)
    >>> f = 440.0     # sound frequency (Hz)
    >>> t = np.linspace(0, T, T*rate, endpoint=False)
    >>> x = (2**23 - 1) * np.sin(2 * np.pi * f * t)
    >>> writewav24("sine24.wav", rate, x)
    """
    a32 = np.asarray(data, dtype=np.int32)
    if a32.ndim == 1:
        # Convert to a 2D array with a single column.
        a32.shape = a32.shape + (1,)
    # By shifting first 0 bits, then 8, then 16, the resulting output
    # is 24 bit little-endian.
    a8 = (a32.reshape(a32.shape + (1,)) >> np.array([0, 8, 16])) & 255
    wavdata = a8.astype(np.uint8).tostring()

    w = wave.open(filename, 'wb')
    w.setnchannels(a32.shape[1])
    w.setsampwidth(3)
    w.setframerate(rate)
    w.writeframes(wavdata)
    w.close()
  