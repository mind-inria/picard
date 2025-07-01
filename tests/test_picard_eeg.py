from eegprep import pop_loadset
from picard import picard
import numpy as np
from scipy.io import loadmat

EEG = loadmat('tests/eeglab_data_with_ica_tmp.set');

picard(EEG['data'].astype(np.float64), ortho=False, verbose=True, whiten=True, random_state=5489) #, w_init=np.eye(32));
