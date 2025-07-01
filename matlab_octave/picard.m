function [Y, W] = picard(X, varargin)
% Runs the Picard algorithm for ICA.
%
% The algorithm is detailed in::
%
%     Pierre Ablin, Jean-Francois Cardoso, and Alexandre Gramfort
%     Faster independent component analysis by preconditioning with Hessian
%     approximations
%     IEEE Transactions on Signal Processing, 2018
%     https://arxiv.org/abs/1706.08171
%
% Picard estimates independent components from the 2-D signal matrix X. The
% rows of X are the input mixed signals. The algorithm outputs [Y, W],
% where Y corresponds to the estimated source matrix, and W to the
% estimated unmixing matrix, such that Y =  W * X.
%
% There are several optional parameters which can be provided in the
% varargin variable.
%
% Optional parameters:
% --------------------
% 'm'                         (int) Size of L-BFGS's memory. Typical values
%                             for m are in the range 3-15.
%                             Default : 7
%
% 'maxiter'                   (int) Maximal number of iterations for the
%                             algorithm.
%                             Default : 100
%
% 'mode'                      (string) Chooses to run the orthogonal
%                             (Picard-O) or unconstrained version of
%                             Picard.
%                             Possible values:
%                             'ortho' (default): runs Picard-O
%                             'standard'       : runs standard Picard
%
% 'tol'                       (float) Tolerance for the stopping criterion.
%                             Iterations stop when the norm of the gradient
%                             gets smaller than tol.
%                             Default: 1e-8
%
% 'lambda_min'                (float) Constant used to regularize the
%                             Hessian approximation. Eigenvalues of the
%                             approximation that are below lambda_min are
%                             shifted to lambda_min.
%                             Default: 1e-2
%
% 'ls_tries'                  (int) Number of tries allowed for the
%                             backtracking line-search. When that
%                             number is exceeded, the direction is thrown
%                             away and the gradient is used instead.
%                             Default: 10
%
% 'whiten'                    (bool) If true, the signals X are whitened
%                             before running ICA. When using Picard-O, the
%                             input signals should be whitened.
%                             Default: true
%
% 'verbose'                   (bool) If true, prints the informations about
%                             the algorithm.
%                             Default: false
%
% 'w_init'                    (matrix) Initial rotation matrix for the
%                             algorithm.
%                             Default: empty (identity matrix)
%
% 'python_defaults'           (bool) If true, uses Python-compatible
%                             defaults for the algorithm.
%                             Default: false
%
% 'distribution'              (string) Distribution used for the
%                             distribution-based ICA.
%                             Possible values:
%                             'logistic' (default)
%                             'logcosh'
%
% 'renormalization'           (string) Renormalization method used for the
%                             distribution-based ICA.
%                             Possible values:
%                             'original' (default)
%                             'pythonlike'
%
% Example:
% --------
%
%  [Y, W] = picard(X, 'mode', 'standard', 'tol', 1e-5)
%
%  [Y, W] = picard(X, 'mode', 'ortho', 'tol', 1e-10, 'verbose', true)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Authors: Pierre Ablin <pierre.ablin@inria.fr>
%          Alexandre Gramfort <alexandre.gramfort@inria.fr>
%          Jean-Francois Cardoso <cardoso@iap.fr>
%          Lukas Stranger <l.stranger@student.tugraz.at>
%
% License: BSD (3-clause)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% First tests

if nargin == 0,
    error('No signal provided');
end

if length(size(X)) > 2,
    error('Input signals should be two dimensional');
end

% if ~isa (X, 'double'),
%   fprintf ('Converting input signals to double...');
%   X = double(X);
% end

[N, T] = size(X);

if N > T,
    error('There are more signals than samples')
end

% Default parameters

m = 7;
maxiter = 100;
mode = 'ortho';
tol = 1e-8;
lambda_min = 0.01;
ls_tries = 10;
whiten = true;
verbose = false;
n_components = size(X, 1);
centering = false;
whitening_mode = 'sph';
w_init = [];
python_defaults = false;
distribution = 'logistic';
renormalization = 'original';

% Read varargin

if mod(length(varargin), 2) == 1,
    error('There should be an even number of optional parameters');
end

for i = 1:2:length(varargin)
    param = lower(varargin{i});
    value = varargin{i + 1};
    switch param
        case 'm'
            m = value;
        case 'maxiter'
            maxiter = value;
        case 'mode'
            mode = value;
        case 'tol'
            tol = value;
        case 'lambda_min'
            lambda_min = value;
        case 'ls_tries'
            ls_tries = value;
        case 'whiten'
            whiten = value;
        case 'pca'
            whitening_mode = 'pca';
            n_components = value;
        case 'centering'
            centering = value;
        case 'verbose'
            verbose = value;
        case 'w_init'
            w_init = value;
        case 'python_defaults'
            python_defaults = value;
        case 'distribution'
            distribution = value;
        case 'renormalization'
            renormalization = value;
        otherwise
            error(['Parameter ''' param ''' unknown'])
    end
end

if python_defaults
    maxiter = 512;
    tol = 1e-7;
    m = 10;
    centering = true;
    distribution = 'logcosh';
    renormalization = 'pythonlike';
    whitening_mode = 'pca';
    if verbose
        disp('Using Python-compatible defaults.');
    end
end

if isempty(w_init)
    if python_defaults
        if verbose, disp('Using random w_init.'); end
        if exist('OCTAVE_VERSION','builtin') ~= 0
            w_init = random_init_octave(n_components); % Random orthogonal matrix
        else
            w_init = random_init(n_components); % Random orthogonal matrix
        end
        w_init = sym_decorrelation(w_init);
    else
        w_init = eye(n_components); % Default identity
    end
end

if whiten == false && n_components ~= size(X, 1),
    error('PCA works only if whiten=true')
end

if n_components ~= getrank(X),
    warning(['Input matrix is of deficient rank. ' ...
            'Please consider to reduce dimensionality (pca) prior to ICA.'])
end

if centering,
   X_mean = mean(X, 2);
   X = X - repmat(X_mean, [1 size(X, 2)]);
end

% Whiten the signals if needed
if whiten,
    [X_white, W_white] = whitening(X, whitening_mode, n_components);
else
    X_white = X;
    W_white = eye(n_components);
end

% Handle w_init (the initial rotation)
if isempty(w_init)
    if python_defaults
        if verbose, disp('Using random w_init.'); end
        w_init = random_init(n_components); % Random orthogonal matrix
    else
        w_init = eye(n_components); % Default identity
    end
end
X_white = w_init * X_white;

% Run ICA
switch mode
    case 'ortho'
        [Y, W_algo] = picardo(X_white, m, maxiter, tol, lambda_min, ls_tries, verbose);
    case 'standard'
        [Y, W_algo] = picard_standard(X_white, m, maxiter, 2, tol, lambda_min, ls_tries, verbose, distribution, renormalization);
    otherwise
        error('Wrong ICA mode')
end

W = W_algo * w_init * W_white;
end

function tmprank2 = getrank(tmpdata)
    % Function originally written in EEGLAB by Arnaud Delorme
    tmprank = rank(tmpdata);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Here: alternate computation of the rank by Sven Hoffman
    %tmprank = rank(tmpdata(:,1:min(3000, size(tmpdata,2)))); old code
    covarianceMatrix = cov(tmpdata', 1);
    [E, D] = eig (covarianceMatrix);
    rankTolerance = 1e-7;
    tmprank2=sum (diag (D) > rankTolerance);
    if tmprank ~= tmprank2
        fprintf('Warning: fixing rank computation inconsistency (%d vs %d) most likely because running under Linux 64-bit Matlab\n', tmprank, tmprank2);
        tmprank2 = min(tmprank, tmprank2);
    end
end

function w_init = random_init(n_components)

    % same random algorithm as in Python
    s = RandStream('mt19937ar', 'Seed', 5489, 'NormalTransform', 'Polar');
    RandStream.setGlobalStream(s);

    % Generate and transpose so positions match NumPy's row-major layout
    w_init = randn(ceil(n_components*n_components/2)*2,1);
    w_init = w_init(:);

    w_initTmp = w_init(1:2:end);
    w_init(1:2:end) = w_init(2:2:end);
    w_init(2:2:end) = w_initTmp;

    w_init = reshape(w_init(1:n_components*n_components), n_components, n_components)';
end

function w_init = random_init_octave(n_components)
    N = ceil(n_components^2/2)*2;
    rand("seed", 5489);
    w = zeros(N,1);
    i = 1;
    while i <= N
        u1 = 2*rand(N,1) - 1;
        u2 = 2*rand(N,1) - 1;
        s  = u1.^2 + u2.^2;
        idx = find(s>0 & s<1);
        r   = sqrt(-2*log(s(idx))./s(idx));
        z1  = u1(idx).*r;
        z2  = u2(idx).*r;
        z   = [z1; z2];
        take = min(length(z), N - i + 1);
        w(i:i+take-1) = z(1:take);
        i = i + take;
    end
    tmp       = w(1:2:end);
    w(1:2:end)= w(2:2:end);
    w(2:2:end)= tmp;
    w_init    = w;
    w_init = reshape(w_init(1:n_components*n_components), n_components, n_components)';
end

function W = sym_decorrelation(W)
    %SYMMETRIC DECORRELATION
    %   W ← (W·Wᵀ)^(-1/2) · W
    
    % Eigen-decompose W·Wᵀ
    [u, S] = eig(W * W.');

    % Extract eigenvalues and form 1./sqrt(s)
    s = diag(S);
    inv_sqrt_s = 1 ./ sqrt(s);

    % Reconstruct (W·Wᵀ)^(-1/2) and apply to W
    W = (u * diag(inv_sqrt_s) * u') * W;
end