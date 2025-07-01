from eegprep import pop_loadset
from eegprep.eeg_picard import eeg_picard
import numpy as np
from scipy.io import loadmat

# EEG = pop_loadset('tests/eeglab_data_with_ica_tmp.set');
EEG = loadmat('tests/eeglab_data_with_ica_tmp.set');

EEG = eeg_picard(EEG, ortho=False, verbose=True, whiten=True) #, w_init=np.eye(32));
