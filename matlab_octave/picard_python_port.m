function [K, W, Y] = picard_python_port(X, varargin)
% PICARD_PYTHON_PORT  Faithful MATLAB port of Python picard solver.
%
% Replicates Python's picard v0.8.2 solver.py + _core_picard.py exactly:
%   - Centering: subtract row means
%   - Whitening: SVD on data (not covariance), K = (u/d)' * sqrt(T)
%   - Core loop: L-BFGS with Tanh density (alpha=1)
%   - Supports standard Picard (ortho=false) and Picard-O (ortho=true)
%   - Supports extended mode (auto sub/super-gaussian detection)
%   - Supports n_components for PCA dimension reduction
%   - Supports identity or random w_init
%
% Usage:
%   [K, W, Y] = picard_python_port(X)
%   [K, W, Y] = picard_python_port(X, 'ortho', true)
%   [K, W, Y] = picard_python_port(X, 'n_components', 30)
%   [K, W, Y] = picard_python_port(X, 'ortho', true, 'extended', true)
%
% Options:
%   'ortho'        - true for Picard-O, false for standard (default: false)
%   'extended'     - true/false/'auto'. 'auto' = same as ortho (default: 'auto')
%   'n_components' - number of components (default: all)
%   'w_init'       - 'identity', 'random', or a matrix (default: 'identity')
%   'max_iter'     - max iterations (default: 512)
%   'tol'          - convergence tolerance (default: 1e-7)
%   'm'            - L-BFGS memory size (default: 10)
%   'lambda_min'   - Hessian regularization (default: 0.01)
%   'ls_tries'     - line search attempts (default: 10)
%   'centering'    - subtract mean (default: true)
%   'verbose'      - print progress (default: true)
%
% Returns:
%   K - whitening matrix [n_components x N]
%   W - unmixing in whitened space [n_components x n_components]
%   Y - estimated sources [n_components x T]
%
% Full unmixing: icaweights = W * K

% Defaults matching Python
opt.max_iter = 512;
opt.tol = 1e-7;
opt.m = 10;
opt.lambda_min = 0.01;
opt.ls_tries = 10;
opt.verbose = true;
opt.centering = true;
opt.ortho = false;
opt.extended = 'auto';
opt.n_components = [];
opt.w_init = 'identity';

% Parse varargin
for i = 1:2:length(varargin)
    opt.(lower(varargin{i})) = varargin{i+1};
end

% Resolve extended default (solver.py line 136-137)
if ischar(opt.extended) && strcmp(opt.extended, 'auto')
    opt.extended = opt.ortho;
end

X = double(X);
[N, T] = size(X);

% Resolve n_components (solver.py line 162-163)
if isempty(opt.n_components)
    opt.n_components = min(N, T);
end
n_comp = opt.n_components;

% --- Step 1: Centering (solver.py line 166-169) ---
if opt.centering
    X_mean = mean(X, 2);
    X = X - repmat(X_mean, 1, T);
end

% --- Step 2: Whitening via SVD on data (solver.py lines 172-185) ---
[u, d_mat, ~] = svd(X, 'econ');
d = diag(d_mat);
K = diag(sqrt(T) ./ d) * u';
K = K(1:n_comp, :);  % truncate to n_components
% enforce fixed-sign for consistency (solver.py lines 178-182, v0.8.2)
for row = 1:size(K, 1)
    [~, j] = max(abs(K(row, :)));
    if K(row, j) < 0
        K(row, :) = -K(row, :);
    end
end
X = K * X;
covariance = eye(n_comp);  % For extended (solver.py line 185)

% --- Step 3: Initialize w_init (solver.py lines 192-201) ---
if ischar(opt.w_init)
    switch opt.w_init
        case 'identity'
            w_init = eye(n_comp);
        case 'random'
            w_init = random_init(n_comp);
            w_init = sym_decorrelation(w_init);
        otherwise
            error('w_init must be ''identity'', ''random'', or a matrix');
    end
else
    w_init = opt.w_init;
    if ~isequal(size(w_init), [n_comp, n_comp])
        error('w_init has invalid shape -- should be [%d x %d]', n_comp, n_comp);
    end
end
X = w_init * X;

% --- Step 4: Core Picard (_core_picard.py) ---
% Tanh density with alpha=1: score=tanh(Y), der=1-tanh(Y)^2

W_algo = eye(n_comp);
Y = X;
s_list = {};
y_list = {};
r_list = {};
signs = ones(n_comp, 1);
current_loss = compute_loss(Y, W_algo, signs, opt.ortho, opt.extended);
G_old = [];
sign_change = false;

if opt.extended
    C = covariance;  % already eye(n_comp)
end

for n_iter = 1:opt.max_iter
    % Score function: Tanh (alpha=1)
    psiY = tanh(Y);
    psidY = 1 - psiY.^2;

    % Relative gradient (_core_picard.py line 90)
    G = (psiY * Y') / T;

    % Squared signals
    Y_square = Y.^2;

    % Extended: kurtosis-based sign estimation (_core_picard.py lines 95-106)
    if opt.extended
        kurt = mean(psidY, 2) .* diag(C) - diag(G);
        signs = sign(kurt);
        if n_iter > 1
            sign_change = any(signs ~= old_signs);
        end
        old_signs = signs;
        G = G .* (signs * ones(1, n_comp));
        psidY = psidY .* (signs * ones(1, T));
        if ~opt.ortho
            G = G + C;
            psidY = psidY + 1;
        end
    end

    % Hessian off-diagonal (_core_picard.py lines 108-111)
    if opt.ortho
        h_off = diag(G);
    else
        h_off = ones(n_comp, 1);
    end

    % Hessian diagonal and regularization (_core_picard.py lines 113-120)
    if opt.ortho
        psidY_mean = mean(psidY, 2);
        diag_mat = psidY_mean * ones(1, n_comp);
        h = 0.5 * (diag_mat + diag_mat' - h_off * ones(1, n_comp) - ones(n_comp, 1) * h_off');
        h(h < opt.lambda_min) = opt.lambda_min;
    else
        h = (psidY * Y_square') / T;
        h = regularize_hessian(h, h_off, opt.lambda_min);
    end

    % Gradient projection (_core_picard.py lines 123-126)
    if opt.ortho
        G = (G - G') / 2;
    else
        G = G - eye(n_comp);
    end

    % Stopping criterion (_core_picard.py line 128)
    gradient_norm = max(abs(G(:)));
    if gradient_norm < opt.tol
        if opt.verbose
            fprintf('Converged at iteration %d, gradient norm = %.6g\n', n_iter, gradient_norm);
        end
        break
    end

    % Update L-BFGS memory (_core_picard.py lines 133-141)
    if n_iter > 1
        s_list{end+1} = direction;
        y_diff = G - G_old;
        y_list{end+1} = y_diff;
        r_list{end+1} = 1.0 / sum(sum(direction .* y_diff));
        if length(s_list) > opt.m
            s_list = s_list(2:end);
            y_list = y_list(2:end);
            r_list = r_list(2:end);
        end
    end
    G_old = G;

    % Flush memory on sign change (_core_picard.py lines 144-146)
    if opt.extended && sign_change
        current_loss = NaN;  % None in Python; recomputed in line search
        s_list = {};
        y_list = {};
        r_list = {};
    end

    % L-BFGS direction (_core_picard.py line 148)
    direction = lbfgs_direction(G, h, h_off, s_list, y_list, r_list, opt.ortho);

    % Line search (_core_picard.py line 151)
    [converged, new_Y, new_W, new_loss, direction] = ...
        line_search_fn(Y, W_algo, direction, signs, current_loss, ...
                       opt.ls_tries, opt.verbose, opt.ortho, opt.extended);

    if ~converged
        direction = -G;
        s_list = {};
        y_list = {};
        r_list = {};
        [~, new_Y, new_W, new_loss, direction] = ...
            line_search_fn(Y, W_algo, direction, signs, current_loss, ...
                           10, false, opt.ortho, opt.extended);
    end

    Y = new_Y;
    W_algo = new_W;
    if opt.extended
        C = W_algo * covariance * W_algo';
    end
    current_loss = new_loss;

    if opt.verbose
        fprintf('iteration %d, gradient norm = %.4g, loss = %.4g\n', ...
            n_iter, gradient_norm, current_loss);
    end
end

% Final W: compose with w_init (solver.py line 215)
W = W_algo * w_init;

end  % picard_python_port


% ===== Helper functions =====

function loss_val = compute_loss(Y, W, signs, ortho, extended)
    % _core_picard.py _loss function
    N = size(Y, 1);
    if ortho
        loss_val = 0;
    else
        loss_val = -log(abs(det(W)));
    end
    % Tanh log_lik: |y| + log1p(exp(-2|y|))
    for k = 1:N
        y = Y(k, :);
        loss_val = loss_val + signs(k) * mean(abs(y) + log1p(exp(-2 * abs(y))));
        if extended && ~ortho
            loss_val = loss_val + 0.5 * mean(y.^2);
        end
    end
end


function h = regularize_hessian(h, h_off, lambda_min)
    % _core_picard.py _regularize_hessian (non-ortho only)
    N = size(h, 1);
    discr = sqrt((h - h').^2 + 4.0 * (h_off * h_off'));
    eigenvalues = 0.5 * (h + h' - discr);
    problematic_locs = eigenvalues < lambda_min;
    problematic_locs(1:(N+1):N*N) = false;  % exclude diagonal
    [i_pb, j_pb] = find(problematic_locs);
    for idx = 1:length(i_pb)
        h(i_pb(idx), j_pb(idx)) = h(i_pb(idx), j_pb(idx)) + ...
            lambda_min - eigenvalues(i_pb(idx), j_pb(idx));
    end
end


function out = solve_hessian(h, h_off, G)
    % _core_picard.py _solve_hessian (non-ortho only)
    det_val = h .* h' - (h_off * h_off');
    out = (h' .* G - (h_off * ones(1, size(G, 2))) .* G') ./ det_val;
end


function direction = lbfgs_direction(G, h, h_off, s_list, y_list, r_list, ortho)
    % _core_picard.py _l_bfgs_direction
    q = G;
    a_list = {};
    for ii = 1:length(s_list)
        s = s_list{end - ii + 1};
        y = y_list{end - ii + 1};
        r = r_list{end - ii + 1};
        alpha = r * sum(sum(s .* q));
        a_list{end+1} = alpha;
        q = q - alpha * y;
    end
    if ortho
        z = q ./ h;
        z = (z - z') / 2;
    else
        z = solve_hessian(h, h_off, q);
    end
    for ii = 1:length(s_list)
        s = s_list{ii};
        y = y_list{ii};
        r = r_list{ii};
        alpha = a_list{end - ii + 1};
        beta = r * sum(sum(y .* z));
        z = z + (alpha - beta) * s;
    end
    direction = -z;
end


function [converged, Y_new, W_new, new_loss, rel_step] = ...
        line_search_fn(Y, W, direction, signs, current_loss, ...
                       ls_tries, verbose, ortho, extended)
    % _core_picard.py _line_search (v0.8.2)
    N = size(W, 1);
    alpha = 1.0;
    if isnan(current_loss)
        current_loss = compute_loss(Y, W, signs, ortho, extended);
    end
    for tmp = 1:ls_tries
        if ortho
            transform = expm(alpha * direction);
        else
            transform = eye(N) + alpha * direction;
        end
        Y_new = transform * Y;
        W_new = transform * W;
        new_loss = compute_loss(Y_new, W_new, signs, ortho, extended);
        if isfinite(new_loss) && new_loss < current_loss
            converged = true;
            rel_step = alpha * direction;
            return
        end
        alpha = alpha / 2.0;
    end
    if verbose
        fprintf('line search failed, falling back to gradient.\n');
    end
    converged = false;
    rel_step = alpha * direction;
end


function w_init = random_init(n_comp)
    % Match Python's default MT19937 random init (from picard.m)
    s = RandStream('mt19937ar', 'Seed', 5489, 'NormalTransform', 'Polar');
    RandStream.setGlobalStream(s);
    w_init = randn(ceil(n_comp * n_comp / 2) * 2, 1);
    w_init = w_init(:);
    w_initTmp = w_init(1:2:end);
    w_init(1:2:end) = w_init(2:2:end);
    w_init(2:2:end) = w_initTmp;
    w_init = reshape(w_init(1:n_comp * n_comp), n_comp, n_comp)';
end


function W = sym_decorrelation(W)
    % _tools.py _sym_decorrelation: W <- (W*W')^{-1/2} * W
    [u, S] = eig(W * W');
    s_vals = diag(S);
    W = (u * diag(1 ./ sqrt(s_vals)) * u') * W;
end
