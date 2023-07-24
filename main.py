import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from scipy import signal

def read_wav_files(filenames):
    data_list = []
    for filename in filenames:
        rate, data = wav.read(filename)
        if len(data.shape) > 1:  # if stereo, take only one channel
            data = data[:, 0]
        data_list.append(data)
    return np.vstack(data_list)

def center(X):
    X = np.array(X)
    mean = X.mean(axis=1, keepdims=True)
    return X - mean

def whitening(X):
    cov = np.cov(X)
    d, E = np.linalg.eigh(cov)
    D = np.diag(d)
    D_inv = np.sqrt(np.linalg.inv(D))
    X_whiten = np.dot(E, np.dot(D_inv, np.dot(E.T, X)))
    return X_whiten

def ICA(X, num_iter=5000, threshold=1e-8):
    n, m = X.shape
    X = center(X)
    X = whitening(X)

    W = np.random.rand(n, n)
    for i in range(num_iter):
        W_old = W
        g = np.tanh(np.dot(W, X))
        g_prime = 1 - g ** 2
        W = np.dot(g, X.T) / m - np.dot(np.diag(g_prime.mean(axis=1)), W)
        W = np.dot(np.linalg.inv(np.sqrt(np.linalg.inv(np.dot(W, W.T)))), W)
        
        if np.max(np.abs(W - W_old)) < threshold:
            break

    S = np.dot(W, X)
    return S

def apply_ICA_and_save(X, filenames):
    W = ICA(X)
    S = np.dot(W, X)
    for i, filename in enumerate(filenames):
        wav.write("separated_" + filename, 16000, S[i])
    return S

def plot_signals(S, S_):
    plt.figure(figsize=(10, 2*len(S)))
    for i, (s, s_) in enumerate(zip(S, S_)):
        plt.subplot(len(S), 1, i+1)
        plt.plot(s, 'b')
        plt.plot(s_, 'r')
    plt.show()

def calculate_correlation_matrix(S, S_):
    return np.corrcoef(S, S_)

if __name__ == "__main__":
    filenames = ["./report2_wav/x1.wav", "./report2_wav/x2.wav", "./report2_wav/x3.wav"]
    X = read_wav_files(filenames)
    S_ = ICA(X)

    original_filenames = ["s1.wav", "s2.wav", "s3.wav"]
    S = read_wav_files(original_filenames)
    correlation_matrix = calculate_correlation_matrix(S, S_)
    print(correlation_matrix)

    plot_signals(S, S_)