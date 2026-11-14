% picard(X, 'extended', true): sub- and super-Gaussian sources are both
% recovered (the logistic density alone cannot separate the sub-Gaussian ones),
% the result is picard_python_port's, and the caller's random state is untouched.
clear
addpath('../matlab_octave/')
rng(0);
T = 20000;
S = [sign(randn(1, T)); rand(1, T) - 0.5; sin(linspace(0, 400, T)); ... % sub-Gaussian
     laplace_rnd([1 T]); laplace_rnd([1 T]).^3];                       % super-Gaussian
S = S ./ std(S, 1, 2);
A = randn(size(S, 1));
X = A * S;

amari = @(P) (sum(sum(abs(P), 2) ./ max(abs(P), [], 2) - 1) + ...
    sum(sum(abs(P), 1) ./ max(abs(P), [], 1) - 1)) / (2 * size(P, 1) * (size(P, 1) - 1));

[~, W_ext] = picard(X, 'mode', 'standard', 'extended', true, 'maxiter', 500, 'tol', 1e-7, 'm', 10);
[~, W_log] = picard(X, 'mode', 'standard', 'maxiter', 500);
[K, W_port] = picard_python_port(X, 'extended', true, 'max_iter', 500, 'tol', 1e-7, 'm', 10, 'verbose', false);
fprintf('Amari index: extended %.4f, logistic %.4f\n', amari(W_ext * A), amari(W_log * A));
assert(amari(W_ext * A) < 0.02, 'extended Picard did not separate the sources');
assert(amari(W_ext * A) < amari(W_log * A), 'extended should beat the logistic density here');
assert(max(abs(W_ext - W_port * K), [], 'all') < 1e-12 * max(abs(W_ext(:))));

state = rng;
picard(X, 'python_defaults', true);
assert(isequal(rng, state), 'picard changed the global random state');
fprintf('test_picard_extended passed\n');
