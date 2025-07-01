EEG = pop_loadset('eeglab_data_with_ica_tmp.set');

[~,whitening_matrix] = whitening(double(EEG.data), 'pca', EEG.nbchan);

whitening_matrix(1:6,1:6)
