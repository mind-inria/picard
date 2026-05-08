function EEG = eeg_picard(EEG, varargin)

[K, W, ~] = picard_python_port(double(EEG.data), 'verbose', true, varargin{:});
EEG.icaweights = W * K;
EEG.icasphere = eye(size(EEG.icaweights, 2));
EEG.icawinv = pinv(EEG.icaweights);
EEG.icachansind = 1:size(EEG.icaweights, 2);
EEG.icaact = EEG.icaweights * EEG.icasphere * EEG.data(EEG.icachansind, :);
EEG.icaact = reshape(EEG.icaact, size(EEG.data, 1), EEG.pnts, EEG.trials);
