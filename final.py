import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mutual_info_score
import seaborn as sns
from mir_eval.separation import bss_eval_sources
from scipy.stats import spearmanr

# load the observed signals
fs, x1 = wav.read('./report2_wav/x1.wav')
_, x2 = wav.read('./report2_wav/x2.wav')
_, x3 = wav.read('./report2_wav/x3.wav')
X = np.array([x1, x2, x3])

# load the source signals
_, s1 = wav.read('./report2_wav/s1.wav')
_, s2 = wav.read('./report2_wav/s2.wav')
_, s3 = wav.read('./report2_wav/s3.wav')
S = np.array([s1, s2, s3])

# normalize the observed signals
X = (X - np.mean(X, axis=1, keepdims=True)) / np.sqrt(np.nanvar(X, axis=1, keepdims=True))

# normalize the source signals
S = (S - np.mean(S, axis=1, keepdims=True)) / np.sqrt(np.nanvar(S, axis=1, keepdims=True))

# whitening of the observed signals
M = X.shape[1]
R = np.dot(X, X.T) / M  # covariance matrix
eig_v, Q = np.linalg.eig(R)
Lambda = np.diag(eig_v)
V = np.dot(np.sqrt(np.linalg.inv(Lambda)), Q.T)
Xh = np.dot(V, X)  # whitened signals

# Implement ICA
def ICA(Xh, max_iter=1000, tol=1e-6):
    m = Xh.shape[0]  # number of sources
    Sh = np.zeros_like(Xh)  # initialize separated signals
    W = np.eye(m)  # initialize unmixing matrix
    W_past = np.eye(m)  # past unmixing matrix

    for i in range(m):
        for _ in range(max_iter):
            W_past[i, :] = W[i, :]
            sh = np.dot(W[i, :].T, Xh)
            w_new = np.mean(np.dot(sh**3, Xh.T)) - 3 * W[i, :]
            if i > 0:
                w_new -= np.dot(np.dot(w_new, W[:i].T), W[:i])
            w_new /= np.sqrt(np.dot(w_new, w_new.T))
            W[i, :] = w_new
            Sh[i, :] = sh  # update separated signal
            if np.max(np.abs(W[i, :] - W_past[i, :])) < tol:
                break
    return Sh

# Reorder seperated signals
def reorder_separated_signals(S, Sh):
    correlation_matrix = np.abs(spearmanr(S.T, Sh.T)[0])
    correlation_matrix = correlation_matrix[:S.shape[0], S.shape[0]:]
    separated_order = np.argmax(correlation_matrix, axis=1)
    return Sh[separated_order]

# Calculate Seperated signals from observed-whitened signals
Sh_initial = ICA(Xh)
Sh = reorder_separated_signals(S, Sh_initial)

# correlation matrix between source signals and seperated signals
plt.figure()  # Create a new figure
corr_matrix = np.abs(np.corrcoef(np.vstack((S, Sh))))  # take absolute value of correlations
labels_sources = ['s1', 's2', 's3']
labels_seperated = ['sh1', 'sh2', 'sh3']
sns.heatmap(corr_matrix, annot=True, fmt=".2f", xticklabels=labels_sources+labels_seperated, yticklabels=labels_sources+labels_seperated)
plt.savefig("final_abs_corr_matrix.png",dpi=500)
# plt.show()

# Correlation matrix between source signals and observed signals
plt.figure()  # Create a new figure
corr_matrix_sources_observed = np.abs(np.corrcoef(np.vstack((S, X))))
labels_observed = ['x1', 'x2', 'x3']
sns.heatmap(corr_matrix_sources_observed, annot=True, fmt=".2f", xticklabels=labels_sources+labels_observed, yticklabels=labels_sources+labels_observed)
plt.savefig("abs_corr_matrix_sources_observed.png",dpi=500)
# plt.show()

# Show original and separated signals spectrum
plt.figure()  # Create a new figure
time = np.arange(S.shape[1]) / fs
fig, axs = plt.subplots(2, 3, figsize=(20, 10))

for i in range(3):
    axs[0, i].plot(time, S[i], label='s'+str(i+1))  # First row for original signals
    axs[0, i].legend()
    axs[0, i].set_xlabel('Time (s)')
    axs[1, i].plot(time, Sh[i], label='sh'+str(i+1))  # Second row for separated signals
    axs[1, i].legend()
    axs[1, i].set_xlabel('Time (s)')

plt.tight_layout()
plt.savefig("final_order_diagram.png",dpi=500)
# plt.show()

# calculate evaluation metrics
def calculate_evaluation_metrics(orig, est):
    # mean squared error
    mse = mean_squared_error(orig, est)
    # mutual information
    mi = mutual_info_score(orig.astype(int), est.astype(int))
    #other metrics
    sdr, sir, sar, _ = bss_eval_sources(orig, est)
    return mse, mi, sdr, sir, sar

# Compute the metrics for the corresponding pairs of source and separated signals
for i in range(3):
    mse, mi, sdr, sir, sar = calculate_evaluation_metrics(S[i], Sh[i])
    print(f'For source signal s{i+1} and separated signal sh{i+1}, MSE: {mse:.2f}, MI: {mi:.2f}, SDR: {sdr[0]:.2f}, SIR: {sir[0]:.2f}, SAR: {sar[0]:.2f}')

# Save the separated signals to wav files
Sh_rescaled = np.int16(Sh/np.max(np.abs(Sh)) * 32767)  # rescale the separated signals
wav.write('./sh1.wav', fs, Sh_rescaled[0])
wav.write('./sh2.wav', fs, Sh_rescaled[1])
wav.write('./sh3.wav', fs, Sh_rescaled[2])
