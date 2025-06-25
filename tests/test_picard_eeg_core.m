EEG = pop_loadset('eeglab_data_with_ica_tmp.set');

EEG = eeg_picard(EEG, 'w_init', eye(32));
