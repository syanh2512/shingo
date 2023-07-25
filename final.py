import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mutual_info_score
import seaborn as sns
from mir_eval.separation import bss_eval_sources

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

# normalize the input signals
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
    W = np.eye(m)  # initialize unmixing matrix
    for _ in range(max_iter):
        W_new = W.copy()
        for i in range(m):
            w = W[i, :]
            sh = np.dot(w.T, Xh)
            w_new = np.mean(np.dot(sh**3, Xh.T)) - 3 * w
            # make w_new orthogonal to the other rows in W
            if i > 0:
                w_new -= np.dot(np.dot(w_new, W[:i].T), W[:i])
            w_new /= np.sqrt(np.dot(w_new, w_new.T))
            W_new[i, :] = w_new
        if np.max(np.abs(W - W_new)) < tol:
            break
        W = W_new
    return W

W = ICA(Xh)
Sh = np.dot(W, Xh)

# correlation matrix
# corr_matrix = np.corrcoef(np.vstack((S, Sh)))
plt.figure()  # Create a new figure
corr_matrix = np.abs(np.corrcoef(np.vstack((S, Sh))))  # take absolute value of correlations

labels = ['s1', 's2', 's3', 'sh1', 'sh2', 'sh3']
sns.heatmap(corr_matrix, annot=True, fmt=".2f", xticklabels=labels, yticklabels=labels)

# sns.heatmap(corr_matrix, annot=True, fmt=".2f")

plt.savefig("final_abs_corr_matrix.png",dpi=500)
# plt.show()

# Correlation matrix between source signals and observed signals
plt.figure()  # Create a new figure
corr_matrix_sources_observed = np.abs(np.corrcoef(np.vstack((S, X))))

labels_sources = ['s1', 's2', 's3']
labels_observed = ['x1', 'x2', 'x3']
sns.heatmap(corr_matrix_sources_observed, annot=True, fmt=".2f", xticklabels=labels_sources+labels_observed, yticklabels=labels_sources+labels_observed)
plt.savefig("abs_corr_matrix_sources_observed.png",dpi=500)

# Show original and separated signals
plt.figure()  # Create a new figure
time = np.arange(S.shape[1]) / fs
fig, axs = plt.subplots(3, 2, figsize=(10, 20))

# Match each sh to its corresponding s
s_order = [0, 1, 2]  # order for s: s3, s2, s1
sh_order = [2, 1, 0]  # order for sh: sh1, sh2, sh3

for i in range(3):
    axs[i, 0].plot(time, S[s_order[i]], label='s'+str(s_order[i]+1))
    axs[i, 0].legend()
    axs[i, 0].set_xlabel('Time (s)')
    axs[i, 1].plot(time, Sh[sh_order[i]], label='sh'+str(sh_order[i]+1))
    axs[i, 1].legend()
    axs[i, 1].set_xlabel('Time (s)')

plt.savefig("final_order_diagram.png",dpi=500)


# plt.show()

# calculate evaluation metrics
def calculate_evaluation_metrics(orig, est):
    # mean squared error
    mse = mean_squared_error(orig, est)
    # mutual information
    mi = mutual_info_score(orig.astype(int), est.astype(int))
    return mse, mi

# Compute the metrics for the corresponding pairs of source and separated signals
for i in range(3):
    mse, mi = calculate_evaluation_metrics(S[s_order[i]], Sh[sh_order[i]])
    sdr, sir, sar, _ = bss_eval_sources(S[s_order[i]], Sh[sh_order[i]])
    print(f'For source signal s{s_order[i]+1} and separated signal sh{sh_order[i]+1}, MSE: {mse:.2f}, MI: {mi:.2f}, SDR: {sdr[0]:.2f}, SIR: {sir[0]:.2f}, SAR: {sar[0]:.2f}')

# Save the separated signals to wav files
Sh_rescaled = np.int16(Sh/np.max(np.abs(Sh)) * 32767)  # rescale the separated signals
wav.write('./sh1.wav', fs, Sh_rescaled[0])
wav.write('./sh2.wav', fs, Sh_rescaled[1])
wav.write('./sh3.wav', fs, Sh_rescaled[2])
