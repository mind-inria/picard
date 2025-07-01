clear
rng(0); % for reproducibility

% Run Python version first so it saves the data.mat array
addpath('../matlab_octave/')

% load the same data
try
   d = load('picard_data.mat','X','A');
catch
   error('Run the python program first');
end
X = d.X;
A = d.A;
% run MATLAB Picard
[Y_mat, W_mat] = picard_standard(X, 10, 200, 2, 1e-6, 0.01, 10, true, 'logcosh', 'pythonlike');
%[Y_mat, W_mat] = picard_standard3(X, 10, 200, 2, 1e-6, 0.01, 10, true);
fprintf('MATLAB finished\n');
