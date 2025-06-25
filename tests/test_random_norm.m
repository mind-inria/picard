
% Seed MT19937 and force the polar Box–Muller method
s = RandStream('mt19937ar', 'Seed', 5489, 'NormalTransform', 'Polar');
RandStream.setGlobalStream(s);

% Generate and transpose so positions match NumPy's row-major layout
n_components = 11;
w_init = randn(ceil(n_components*n_components/2)*2,1);
w_init = w_init(:);

w_initTmp = w_init(1:2:end);
w_init(1:2:end) = w_init(2:2:end);
w_init(2:2:end) = w_initTmp;

w_init = reshape(w_init(1:n_components*n_components), n_components, n_components)';


w_init
