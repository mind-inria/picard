function [K, W, Y] = picard_python_port(X, varargin)
% PICARD_PYTHON_PORT  Faithful MATLAB port of Python picard solver.
%
% Replicates Python's picard v0.8.2 solver.py + _core_picard.py exactly:
%   - Centering: subtract row means
%   - Whitening: SVD on data (not covariance), K = (u/d)' * sqrt(T)
%   - w_init: identity (deterministic)
%   - Core loop: L-BFGS with Tanh density (alpha=1)
%   - Supports both standard Picard (ortho=false) and Picard-O (ortho=true)
%
% Usage:
%   [K, W, Y] = picard_python_port(X)
%   [K, W, Y] = picard_python_port(X, 'ortho', true)
%   [K, W, Y] = picard_python_port(X, 'max_iter', 512, 'tol', 1e-7, ...)
%
% Returns:
%   K - whitening matrix [N x N]
%   W - unmixing in whitened space [N x N]
%   Y - estimated sources [N x T]
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

% Parse varargin
for i = 1:2:length(varargin)
    opt.(lower(varargin{i})) = varargin{i+1};
end

X = double(X);
[N, T] = size(X);

% --- Step 1: Centering (solver.py line 170) ---
if opt.centering
    X_mean = mean(X, 2);
    X = X - repmat(X_mean, 1, T);
end

% --- Step 2: Whitening via SVD on data (solver.py lines 172-177) ---
[u, d_mat, ~] = svd(X, 'econ');
d = diag(d_mat);
K = diag(sqrt(T) ./ d) * u';
K = K(1:N, :);
% enforce fixed-sign for consistency (solver.py lines 178-182, v0.8.2)
% Flip each row of K so max-abs element is positive.
for row = 1:size(K, 1)
    [~, j] = max(abs(K(row, :)));
    if K(row, j) < 0
        K(row, :) = -K(row, :);
    end
end
X = K * X;

% --- Step 3: Apply w_init = I (solver.py line 199) ---
w_init = eye(N);
X = w_init * X;

% --- Step 4: Core Picard (_core_picard.py) ---
% Tanh density with alpha=1: score=tanh(Y), der=1-tanh(Y)^2

W_algo = eye(N);
Y = X;
s_list = {};
y_list = {};
r_list = {};
current_loss = compute_loss(Y, W_algo, opt.ortho);
G_old = [];

for n_iter = 1:opt.max_iter
    % Score function: Tanh (alpha=1)
    psiY = tanh(Y);
    psidY = 1 - psiY.^2;

    % Relative gradient (_core_picard.py line 90)
    % np.inner(psiY, Y) for 2D = psiY @ Y.T
    G = (psiY * Y') / T;

    % Hessian off-diagonal (_core_picard.py lines 108-111)
    if opt.ortho
        h_off = diag(G);
    else
        h_off = ones(N, 1);
    end

    % Hessian diagonal and regularization (_core_picard.py lines 113-120)
    if opt.ortho
        % Ortho: symmetric Hessian from mean of psidY
        psidY_mean = mean(psidY, 2);
        diag_mat = psidY_mean * ones(1, N);
        h = 0.5 * (diag_mat + diag_mat' - h_off * ones(1, N) - ones(N, 1) * h_off');
        h(h < opt.lambda_min) = opt.lambda_min;
    else
        % Non-ortho: Hessian from inner product
        Y_square = Y.^2;
        h = (psidY * Y_square') / T;
        h = regularize_hessian(h, h_off, opt.lambda_min);
    end

    % Gradient projection (_core_picard.py lines 123-126)
    if opt.ortho
        G = (G - G') / 2;
    else
        G = G - eye(N);
    end

    % Stopping criterion (_core_picard.py line 128)
    gradient_norm = max(abs(G(:)));
    if gradient_norm < opt.tol
        if opt.verbose
            fprintf('Converged at iteration %d, gradient norm = %.6g\n', n_iter, gradient_norm);
        end
        break
    end

    % Update L-BFGS memory (_core_picard.py lines 131-138)
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

    % L-BFGS direction (_core_picard.py line 148)
    direction = lbfgs_direction(G, h, h_off, s_list, y_list, r_list, opt.ortho);

    % Line search (_core_picard.py line 151)
    [converged, new_Y, new_W, new_loss, direction] = ...
        line_search_fn(Y, W_algo, direction, current_loss, opt.ls_tries, opt.verbose, opt.ortho);

    if ~converged
        direction = -G;
        s_list = {};
        y_list = {};
        r_list = {};
        [~, new_Y, new_W, new_loss, direction] = ...
            line_search_fn(Y, W_algo, direction, current_loss, 10, false, opt.ortho);
    end

    Y = new_Y;
    W_algo = new_W;
    current_loss = new_loss;

    if opt.verbose
        fprintf('iteration %d, gradient norm = %.4g, loss = %.4g\n', ...
            n_iter, gradient_norm, current_loss);
    end
end

% Final W: compose with w_init (_core_picard.py line 209 in solver.py)
W = W_algo * w_init;

end  % picard_python_port


% ===== Helper functions =====

function loss_val = compute_loss(Y, W, ortho)
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
        loss_val = loss_val + mean(abs(y) + log1p(exp(-2 * abs(y))));
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
        line_search_fn(Y, W, direction, current_loss, ls_tries, verbose, ortho)
    % _core_picard.py _line_search (v0.8.2)
    N = size(W, 1);
    alpha = 1.0;
    for tmp = 1:ls_tries
        if ortho
            transform = expm(alpha * direction);
        else
            transform = eye(N) + alpha * direction;
        end
        Y_new = transform * Y;
        W_new = transform * W;
        new_loss = compute_loss(Y_new, W_new, ortho);
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
