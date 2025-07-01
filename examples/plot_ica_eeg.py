"""
=======================================================
Blind source separation using preconditioned ICA on EEG
=======================================================

The example runs the Picard-O algorithm proposed in:

Pierre Ablin, Jean-François Cardoso, Alexandre Gramfort
"Faster ICA under orthogonal constraint"
ICASSP, 2018
https://arxiv.org/abs/1711.10873

"""  # noqa

# Author: Pierre Ablin <pierre.ablin@inria.fr>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.datasets import sample
from scipy.stats import kurtosis

from picard import picard

print(__doc__)

###############################################################################
# Generate sample EEG data
data_path = sample.data_path()
raw_fname = data_path / 'MEG/sample/sample_audvis_filt-0-40_raw.fif'

raw = mne.io.read_raw_fif(raw_fname, preload=True)
raw.filter(1, 40, n_jobs=1)  # 1Hz high pass is often helpful for fitting ICA

picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False,
                       stim=False, exclude='bads')

random_state = 0
data = raw[picks, :][0]
data = data[:, ::2]  # decimate a bit

###############################################################################
# Run ICA on data, after reducing the dimension

K, W, Y = picard(data, n_components=30, ortho=True, random_state=0)

###############################################################################
# Plot results

n_plots = 10
T_plots = 1000
order = np.argsort(kurtosis(Y[:, :T_plots], axis=1))[::-1]
models = [data[:n_plots], Y[order[:n_plots][::-1]]]
names = ['Observations (raw EEG)',
         'ICA recovered sources']

fig, axes = plt.subplots(2, 1, figsize=(7, 7))
for ii, (model, name, ax) in enumerate(zip(models, names, axes)):
    ax.set_title(name)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    offsets = np.max(model, axis=1) - np.min(model, axis=1)
    offsets = np.cumsum(offsets)
    ax.plot((model[:, :T_plots] + offsets[:, np.newaxis]).T, 'k')

fig.tight_layout()
plt.show()
